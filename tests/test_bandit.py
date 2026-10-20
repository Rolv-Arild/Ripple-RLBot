import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rlbot.flat import MatchPhase as Phase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent import Agent, ActionMode
from bandit import BANDIT_MODES, EpisodeSelector, ModeBandit
from packets import field_info, packet


class BanditTests(unittest.TestCase):
    def make_bandit(self, *args, **kwargs):
        bandit = ModeBandit(*args, **kwargs)
        self.addCleanup(bandit.close)
        return bandit

    def test_legacy_hierarchical_prior_and_perfect_success_rule(self):
        bandit = self.make_bandit('v1.1', seed=73)
        exact = bandit.context(['Ripple', 'Ally'], ['A', 'B'])
        same_size = bandit.context(['Ripple', 'Other ally'], ['C', 'D'])
        other_size = bandit.context(['Ripple'], ['A'])
        for context, outcomes in ((other_size, [1, 1, 1, -1]),
                                  (same_size, [1, -1, -1]),
                                  (exact, [1, 1, -1, -1, -1])):
            for outcome in outcomes:
                bandit.record(context, 'kl', outcome)
        bandit.record(exact, 'stochastic', -1)
        bandit.record(exact, 'inverse-kl', 0)
        # Legacy priors: rest=3.5/5=.7; size=(1+.7)/4=.425.
        alpha, beta, _ = bandit.posterior(exact)
        np.testing.assert_allclose(alpha, [2.425, .5, .5, .5])
        np.testing.assert_allclose(beta, [3.575, .5, 1.5, .5])
        expected = BANDIT_MODES[np.random.default_rng(73).beta(alpha, beta).argmax()]
        self.assertEqual(bandit.select(exact), expected)
        bandit.record(exact, 'deterministic', 1)
        self.assertEqual(bandit.select(exact), ActionMode.DETERMINISTIC)
        bandit.record(exact, 'inverse-kl', 1)
        # First perfect mode in the old config's candidate order wins ties.
        self.assertEqual(bandit.select(exact), ActionMode.INVERSE_KL)
        self.assertEqual(bandit.context(['Ally', 'Ripple'], ['B', 'A']), exact)

    def test_persistence_and_version_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bandit.sqlite3'
            first = ModeBandit('v1.1', path)
            context = first.context(['Ripple'], ['Nexto'])
            first.record(context, 'deterministic', 1)
            first.close()
            for profile, expected in (('v1.1', 1), ('v1.0', 0)):
                bandit = ModeBandit(profile, path)
                self.assertEqual(bandit.statistics(context)[0, 0].sum(), expected)
                bandit.close()

    def test_delayed_scores_and_replay_credit_once_for_both_colors(self):
        sequence = [(100, Phase.Active, (0, 0)),
                    (108, Phase.GoalScored, (0, 0)),
                    (112, Phase.Replay, (1, 0)),
                    (116, Phase.Replay, (1, 0)),
                    (120, Phase.Countdown, (1, 0)),
                    (124, Phase.Kickoff, (1, 0)),
                    (128, Phase.Active, (1, 0)),
                    (132, Phase.GoalScored, (1, 0)),
                    (136, Phase.Replay, (1, 1)),
                    (140, Phase.Countdown, (1, 1))]
        for team in (0, 1):
            bandit = self.make_bandit(f'color-{team}', seed=3)
            selector = EpisodeSelector(bandit)
            modes, outcomes = [], []
            with patch.object(bandit, 'record', wraps=bandit.record) as record:
                for frame, phase, scores in sequence:
                    modes.append(selector.update(packet(frame, phase, scores=scores), team))
                for call in record.call_args_list:
                    outcomes.append(call.args[2])
                self.assertEqual(outcomes, [1, -1] if team == 0 else [-1, 1])
                self.assertEqual(record.call_args_list[0].args[1], modes[0])
                self.assertEqual(record.call_args_list[1].args[1], modes[5])
                np.testing.assert_array_equal(selector.results, [1, 1, 0])

    def test_pause_restart_and_roster_changes_are_not_losses(self):
        bandit = self.make_bandit('restart')
        selector = EpisodeSelector(bandit)
        with patch.object(bandit, 'record', wraps=bandit.record) as record:
            for frame, phase in ((100, Phase.Kickoff), (104, Phase.Paused),
                                 (500, Phase.Active), (2, Phase.Active)):
                selector.update(packet(frame, phase), 0)
            selector.update(packet(6, count=4), 0)
            self.assertEqual(record.call_count, 0)
            selector.update(packet(10, Phase.Ended, count=4), 0)
            selector.update(packet(14, Phase.Ended, count=4), 0)
            self.assertEqual(record.call_count, 1)
            self.assertEqual(record.call_args.args[2], 0)

    def test_score_preceding_goal_phase_does_not_create_a_second_episode(self):
        bandit = self.make_bandit('early')
        selector = EpisodeSelector(bandit)
        for frame, phase, scores in ((100, Phase.Active, (0, 0)),
                                     (104, Phase.Active, (1, 0)),
                                     (108, Phase.Active, (1, 0)),
                                     (112, Phase.GoalScored, (1, 0)),
                                     (116, Phase.Countdown, (1, 0)),
                                     (120, Phase.Countdown, (1, 0))):
            selector.update(packet(frame, phase, scores=scores), 0)
        np.testing.assert_array_equal(selector.results, [1, 0, 0])

    def test_adaptive_warmup_covers_both_networks_and_restores_mode(self):
        with patch('agent.OnnxModel') as factory:
            factory.return_value.return_value = np.zeros((2, 276), dtype=np.float32)
            agent = Agent(Path('policy.onnx'), Path('bcm.onnx'), backend='onnx',
                          available_modes=BANDIT_MODES)
            self.assertEqual(factory.call_count, 2)
            agent.warmup(2)
            self.assertEqual(agent.mode, ActionMode.STOCHASTIC)
            # Both primary and BCM calls were warmed before any packet.
            self.assertEqual(factory.return_value.call_count, 18)

    def test_runtime_bandit_switches_and_fixed_mode_has_no_database(self):
        from bot import Ripple
        for name in ('v1.0', 'v1.1'):
            bot = Ripple(variant=name, seed=7, stochastic_kickoffs=False)
            self.addCleanup(bot.retire)
            bot.indices, bot.team, bot.field_info = [0], 0, field_info()
            bot.initialize()
            with patch.object(bot.selector.bandit, 'select', side_effect=[ActionMode.KL, ActionMode.STOCHASTIC]):
                bot.get_outputs(packet(100, Phase.Kickoff))
                self.assertEqual(bot.agent.mode, ActionMode.KL)
                bot.get_outputs(packet(104, Phase.GoalScored, scores=(1, 0)))
                bot.get_outputs(packet(108, Phase.Countdown, scores=(1, 0)))
                self.assertEqual(bot.agent.mode, ActionMode.STOCHASTIC)
                bot.get_outputs(packet(112, Phase.Kickoff, scores=(1, 0)))
                np.testing.assert_array_equal(bot.selector.results, [1, 0, 0])
            with patch('bot.ModeBandit') as database:
                fixed = Ripple(variant=name, mode='stochastic')
                self.assertIsNone(fixed.selector)
                self.assertIsNone(fixed.agent.base)
                database.assert_not_called()

    def test_status_is_once_per_wall_clock_minute_including_pauses(self):
        from bot import Ripple
        with patch('bot.perf_counter', return_value=100.) as clock:
            bot = Ripple(mode='stochastic')
            bot.indices, bot.team, bot.field_info = [0], 0, field_info()
            with patch.object(bot, '_logger') as logger:
                bot.initialize()
                self.assertEqual(logger.info.call_count, 1)  # Startup only.
                for frame in range(100, 800, 4):
                    clock.return_value = 159.
                    bot.get_outputs(packet(frame))
                self.assertEqual(logger.info.call_count, 1)  # Not decision-count based.
                clock.return_value = 160.
                bot.get_outputs(packet(800, Phase.Paused))
                self.assertEqual(logger.info.call_count, 2)
                self.assertIn('p95=', logger.info.call_args.args[9])
                self.assertEqual(len(bot.inference_ms), 0)
                for timestamp in (161., 190., 219.999):
                    clock.return_value = timestamp
                    bot.get_outputs(packet(804, Phase.Paused))
                self.assertEqual(logger.info.call_count, 2)
                clock.return_value = 220.
                bot.get_outputs(packet(808, Phase.Paused))
                self.assertEqual(logger.info.call_count, 3)
                self.assertIn('no decisions', logger.info.call_args.args)
