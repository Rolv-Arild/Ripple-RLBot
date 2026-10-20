import os
import hashlib
import sys
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent import Agent, OnnxModel, log_softmax
from variants import ASSETS, VARIANTS, model_paths, select_variant, source_models
from obs import RippleObs
from test_port import state_fixture, torch, requires_originals


def observation_corpus(samples=12):
    rng = np.random.default_rng(823)
    for count in (2,4,6):
        for step in range(samples):
            state = state_fixture(count,kickoff=step==0)
            state.ball.position += rng.uniform(-500,500,3) if step else 0
            state.ball.position[2] = abs(state.ball.position[2])
            for p in state.players:
                p.car_data.position += rng.uniform(-1000,1000,3)
                p.car_data.position[2] = abs(p.car_data.position[2])
                p.boost_amount = rng.uniform(0,1)
                p.on_ground = step%3==0
            queue = rng.uniform(-1,1,(6,count,8))
            obs = RippleObs(queue)
            obs.reset(state)
            # Include a genuine reset observation, then ordinary action history.
            if step > 0:
                queue[:] = rng.uniform(-1,1,queue.shape)
                queue[:,:,5:] = rng.integers(0,2,queue[:,:,5:].shape)
            obs.pre_step(state,controlled_count=count//2)
            yield obs.current_obs


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = list(observation_corpus())

    def test_variants_and_bob_configs(self):
        bob = tomllib.loads((ROOT/'bob.toml').read_text())
        configs = bob['config'][0]['bot_configs']
        self.assertEqual(len(configs),2)
        self.assertEqual(len({Path(config).name for config in configs}), 2)
        outputs = set()
        for config in configs:
            settings = tomllib.loads((ROOT/config).read_text())['settings']
            with patch.dict(os.environ,{'RLBOT_AGENT_ID':settings['agent_id']}):
                variant = select_variant()
                policy, base = model_paths(variant,'onnx')
                self.assertTrue(policy.is_file())
                self.assertTrue((ROOT/config).parent.joinpath(settings['logo_file']).is_file())
                for key in ('logo_file', 'loadout_file'):
                    self.assertNotIn(settings[key], outputs)
                    outputs.add(settings[key])

    @requires_originals
    def test_onnx_matches_original_models_on_observations(self):
        for path in source_models():
            name = path.relative_to(ASSETS).with_suffix('').as_posix()
            original = torch.jit.load(str(path),map_location='cpu').eval()
            onnx = OnnxModel(path.with_suffix('.onnx'))
            # The 1M imitation model emits unusually large logits; float32
            # rounding is amplified by softmax. Bound probability drift as well
            # as requiring identical greedy actions. The RL policies and 6M
            # imitation model have much tighter probability agreement.
            probability_tolerance = 5e-3 if name==VARIANTS['v1.1'].base else 1e-4
            with torch.inference_mode():
                for obs in self.corpus:
                    expected = original(tuple(torch.from_numpy(x) for x in obs)).numpy()
                    actual = onnx(obs)
                    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)
                    np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
                    expected_p, actual_p = np.exp(log_softmax(expected)), np.exp(log_softmax(actual))
                    np.testing.assert_allclose(actual_p,expected_p,rtol=0,atol=probability_tolerance)

    def test_export_metadata_matches_source(self):
        for path in source_models():
            model = OnnxModel(path.with_suffix('.onnx'))
            metadata = model.session.get_modelmeta().custom_metadata_map
            self.assertRegex(metadata['source_sha256'], r'^[0-9a-f]{64}$')
            if path.is_file():
                self.assertEqual(metadata['source_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(metadata['precision'],'float32')

    @requires_originals
    def test_action_modes_agree_between_backends(self):
        for variant in VARIANTS.values():
            for mode in ('stochastic','deterministic','kl','inverse-kl'):
                if variant.base is None and mode in ('kl','inverse-kl'):
                    continue
                agents = []
                for backend in ('onnx','torch'):
                    policy, base = model_paths(variant,backend)
                    agents.append(Agent(policy,base if mode in ('kl','inverse-kl') else None,
                                        mode=mode,backend=backend,seed=93))
                for obs in self.corpus:
                    np.testing.assert_array_equal(agents[0].act(obs),agents[1].act(obs))

    def test_bcm_seed_controls(self):
        from bot import Ripple
        from packets import field_info,packet
        for name, variant in VARIANTS.items():
            policy, base = model_paths(variant, 'onnx', bcm_only=True)
            self.assertEqual(policy, model_paths(variant, 'onnx')[1])
            self.assertIsNone(base)
            bot = Ripple(variant=name, bcm_only=True)
            bot.indices = [0]
            bot.field_info = field_info()
            bot.initialize()
            bot.get_outputs(packet(100))
            np.testing.assert_array_equal(bot.queue[1:,0,[0,6]],1.)

    def test_bcm_only_loads_no_rl_network(self):
        from bot import Ripple
        for name, variant in VARIANTS.items():
            with patch('agent.OnnxModel') as factory:
                bot = Ripple(variant=name, bcm_only=True)
                factory.assert_called_once_with(model_paths(variant, 'onnx')[1])
                self.assertIsNone(bot.agent.base)
            with self.assertRaisesRegex(ValueError, 'BCM-only'):
                Ripple(variant=name, bcm_only=True, mode='kl')

    def test_single_network_modes_do_not_load_base(self):
        policy, base = model_paths(VARIANTS['v1.1'], 'onnx')
        for mode in ('stochastic', 'deterministic'):
            with patch('agent.OnnxModel') as factory:
                agent = Agent(policy, base, mode=mode, backend='onnx')
                factory.assert_called_once_with(policy)
                self.assertIsNone(agent.base)


if __name__ == '__main__':
    unittest.main()
