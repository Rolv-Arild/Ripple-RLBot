import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from act import RippleAction, RippleLookupAction
from agent import Agent, ActionMode, load_model
from obs import RippleObs, BOOST_LOCATIONS
from timing import DecisionClock
from variants import VARIANTS, model_paths, source_models
from rlbot import flat
from rlgym_compat import V1PlayerData
import RocketSim as rs

try:
    from torch_runtime import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = None

requires_originals = unittest.skipUnless(
    torch is not None and all(path.is_file() for path in source_models()),
    'Optional comparison: restore .pt exports and install the export dependency group')


def state_fixture(count=6, kickoff=False):
    rng = np.random.default_rng(53)
    players = []
    for i in range(count):
        p = V1PlayerData()
        p.player_id = 10 + i
        p.team_num = i % 2
        p.car_data.position = rng.uniform(-2000, 2000, 3)
        p.car_data.linear_velocity = np.zeros(3) if kickoff else rng.uniform(-1500, 1500, 3)
        p.car_data.angular_velocity = rng.uniform(-3, 3, 3)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q[:, 2] *= np.linalg.det(q)
        p.car_data._rotation_mtx = q
        p.car_data._has_computed_rot_mtx = True
        p.on_ground = True
        p.has_jump = p.has_flip = True
        p.boost_amount = 0.5
        players.append(p)
    return SimpleNamespace(players=players, boost_pads=np.ones(34), ball=SimpleNamespace(
        position=np.array([0., 0., 93.]) if kickoff else np.array([300., 700., 500.]),
        linear_velocity=np.array([0., 0., 0.]) if kickoff else np.array([100., 200., -100.]),
        angular_velocity=np.array([0., 0., 0.])))


def reference_class(path, name, **namespace):
    """Run the real reference source, stripping only unavailable training imports."""
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    scope = dict(np=np, rs=rs, Any=object, GameState=object, PlayerData=object,
                 ObsBuilder=object, Space=object, ORANGE_TEAM=1,
                 BOOST_LOCATIONS=BOOST_LOCATIONS, TICK_SKIP=4, TICKS_PER_SECOND=120,
                 INVERT_SIDE_ACTIONS=np.array([1,-1,1,-1,-1,1,1,1]), **namespace)
    exec(compile(tree, str(path), "exec"), scope)
    return scope[name]


class PortTests(unittest.TestCase):
    def test_real_packet_pipeline_and_unsorted_mapping(self):
        from bot import Ripple
        from packets import packet, field_info
        bot = Ripple(mode='deterministic', stochastic_kickoffs=False)
        bot.indices = [2, 0]
        bot.team = 0
        bot.field_info = field_info()
        bot.initialize()
        bot.agent.act = lambda obs, stochastic=False: np.array([1, 2])
        bot.get_outputs(packet(90, flat.MatchPhase.Countdown))
        for frame in range(100, 124):
            controls = bot.get_outputs(packet(frame, flat.MatchPhase.Kickoff))
            self.assertEqual(set(controls), {0, 2})
            self.assertEqual(controls[0].throttle, 0.)
        controls = bot.get_outputs(packet(124, flat.MatchPhase.Kickoff))
        lut = RippleLookupAction().lookup_table
        self.assertEqual(controls[0].steer, lut[1,1])
        self.assertEqual(controls[0].handbrake, bool(lut[1,7]))
        self.assertEqual(controls[2].handbrake, bool(lut[2,7]))
        steps = bot.obs.steps
        bot.get_outputs(packet(137))
        self.assertEqual(bot.obs.steps, steps + 1)
        self.assertEqual(bot.clock.next_frame, 140)
        self.assertEqual(bot.missed_decisions, 2)
        bot.get_outputs(packet(140, count=4))
        self.assertEqual(len(bot.obs.demo_timers), 4)
        bot.get_outputs(packet(141, flat.MatchPhase.Replay, count=4))
        bot.get_outputs(packet(142, flat.MatchPhase.Countdown, count=4))
        self.assertTrue(np.all(bot.queue == 0))
        bot.get_outputs(packet(143, flat.MatchPhase.Active, count=4))
        bot.get_outputs(packet(144, flat.MatchPhase.Paused, count=4))
        misses = bot.missed_decisions
        bot.get_outputs(packet(500, flat.MatchPhase.Active, count=4))
        self.assertEqual(bot.missed_decisions, misses)
        bot.get_outputs(packet(2, flat.MatchPhase.Active, count=4))
        self.assertEqual(bot.last_decision, 2)

    def test_clock_keeps_phase_after_missing_packets(self):
        clock = DecisionClock()
        self.assertEqual([clock.advance(f) for f in [100, 101, 104, 113, 114, 116]],
                         [1, 0, 1, 2, 0, 1])
        self.assertEqual(clock.advance(2), 1)

    def test_delay_and_nan_reset(self):
        queue = np.zeros((6, 2, 8))
        parser = RippleAction(queue)
        for _ in range(6):
            np.testing.assert_array_equal(parser.parse_actions(np.array([1, 2]), None), 0)
        np.testing.assert_array_equal(parser.parse_actions(np.array([3, 4]), None),
                                      parser.parser.lookup_table[[1, 2]])
        queue[:, 0] = np.nan
        result = parser.parse_actions(np.array([5, 6]), None)
        np.testing.assert_array_equal(result[0], parser.parser.lookup_table[5])

    @requires_originals
    def test_action_table_matches_model(self):
        policy, _ = model_paths(VARIANTS['v1.1'], 'torch')
        model = torch.jit.load(str(policy), map_location='cpu')
        np.testing.assert_allclose(RippleLookupAction().lookup_table,
                                   model.state_dict()['net.lookup_table'].numpy(), atol=1e-7)

    @unittest.skipUnless((ROOT / 'external/Ripple/configuration/observation_builder2.py').exists(),
                         'Training source is an optional external reference')
    def test_training_observation_parity(self):
        reference = reference_class(ROOT / 'external/Ripple/configuration/observation_builder2.py', 'RippleObs')
        for count in (2, 4, 6):
            for kickoff in (False, True):
                state = state_fixture(count, kickoff)
                q_ref, q_new = np.zeros((6, 6, 8)), np.zeros((6, count, 8))
                old, new = reference(q_ref), RippleObs(q_new)
                old.reset(state)
                new.reset(state)
                for step in range(8):
                    old.pre_step(state)
                    new.pre_step(state, controlled_count=min(count, 3))
                    for a, b in zip(old.current_obs, new.current_obs):
                        np.testing.assert_allclose(a[:min(count, 3)], b, rtol=2e-6, atol=2e-7)
                    action = RippleLookupAction().lookup_table[(step * 29) % 276]
                    q_ref[:] = action
                    q_new[:] = action
                    for i, player in enumerate(state.players):
                        player.on_ground = step < 2 or step > 5
                        old.build_obs(player, state, action)
                        if i < min(count, 3):
                            new.build_obs(player, state, action)

    @requires_originals
    def test_model_optimization_and_mirroring(self):
        rng = np.random.default_rng(20)
        for path in model_paths(VARIANTS['v1.1'], 'torch'):
            original = torch.jit.load(str(path), map_location='cpu').eval()
            optimized = load_model(path)
            for batch in (1, 2, 3):
                arrays = [rng.normal(size=s).astype(np.float32) for s in
                          ((batch,32), (batch,5,21), (batch,6,9), (batch,6,8), (batch,34,5))]
                arrays[0][:,30] = np.arange(batch) % 2
                arrays[1][:,3:,15] = 0  # Include masked players.
                inputs = tuple(torch.from_numpy(x) for x in arrays)
                with torch.inference_mode():
                    expected, actual = original(inputs), optimized(inputs)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
                np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
                self.assertEqual(actual.shape, (batch, 276))

    def test_ball_cache_reuses_training_tolerance(self):
        state = state_fixture()
        obs = RippleObs(np.zeros((6,6,8)))
        obs.reset(state)
        obs.pre_step(state, 1)
        predicted = obs.ball_predictions[1].copy()
        state.ball.position = predicted[:3] + 0.1
        state.ball.linear_velocity = predicted[3:6]
        state.ball.angular_velocity = predicted[6:]
        count = obs.ball_pred_resets
        obs.pre_step(state, 1)
        self.assertEqual(obs.ball_pred_resets, count)

    @requires_originals
    def test_original_base_weight_loading_matches_direct_export(self):
        policy_path, base_path = model_paths(VARIANTS['v1.1'], 'torch')
        policy = torch.jit.load(str(policy_path), map_location='cpu').eval()
        base = torch.jit.load(str(base_path), map_location='cpu').eval()
        policy.load_state_dict({'net.' + k: v for k,v in base.state_dict().items()})
        for batch in (1, 3):
            obs = tuple(torch.zeros(shape) for shape in ((batch,32), (batch,5,21),
                        (batch,6,9), (batch,6,8), (batch,34,5)))
            with torch.inference_mode():
                torch.testing.assert_close(policy(obs), base(obs))


if __name__ == '__main__':
    unittest.main()
