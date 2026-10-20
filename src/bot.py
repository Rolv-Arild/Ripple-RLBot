"""RLBot v5 hivemind adapter for Ripple's four-tick delayed policy."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from rlbot.flat import ControllerState, MatchPhase
from rlbot.managers.hivemind import Hivemind
from rlgym_compat import V1GameState

from act import RippleAction
from agent import Agent, ActionMode
from bandit import BANDIT_MODES, EpisodeSelector, ModeBandit
from obs import RippleObs
from timing import DecisionClock
from variants import VARIANTS, select_variant, model_paths

class Ripple(Hivemind):
    def __init__(self, model=None, base=None,
                 mode=None, stochastic_kickoffs=True,
                 variant=None, backend='onnx', seed=None, bcm_only=False,
                 bandit_result_file=None):
        self.variant = select_variant(variant)
        self.bcm_only = bcm_only
        mode = mode or ('stochastic' if bcm_only else 'bandit')
        adaptive = mode == 'bandit'
        if bcm_only and (adaptive or ActionMode(mode) in (ActionMode.KL, ActionMode.INVERSE_KL)):
            raise ValueError('BCM-only mode supports stochastic or deterministic selection')
        super().__init__(self.variant.agent_id)
        default_model, default_base = model_paths(self.variant,backend,bcm_only=bcm_only)
        model = model or default_model
        if adaptive or ActionMode(mode) in (ActionMode.KL, ActionMode.INVERSE_KL):
            base = base or default_base
        self.agent = Agent(model, base, BANDIT_MODES[0] if adaptive else mode,
                           backend=backend, seed=seed,
                           available_modes=BANDIT_MODES if adaptive else None)
        profile = self.variant.agent_id
        if Path(model) != default_model or base is not None and Path(base) != default_base:
            profile += f'|{Path(model).resolve()}|{Path(base).resolve() if base else ""}'
        self.selector = EpisodeSelector(ModeBandit(profile, bandit_result_file, seed)) if adaptive else None
        self.bandit_result_file = bandit_result_file
        self.stochastic_kickoffs = stochastic_kickoffs
        self.clock = DecisionClock()
        self.signature = None
        self.previous_phase = None
        self.last_decision = None
        self.last_packet_frame = None
        self.inference_ms = []
        self.missed_decisions = 0
        self.slow_decisions = 0
        self.controls = {}
        self.last_status_time = perf_counter()

    def initialize(self):
        self.game_state = V1GameState(self.field_info, self.match_config, tick_skip=4)
        self.agent.warmup(len(self.indices))
        self.last_status_time = perf_counter()
        selection = 'bandit' if self.selector else self.agent.mode.value
        self._logger.info('%s%s ready: %s, %d car(s), selection=%s, bandit history=%s; status every 60s',
                          self.variant.name, ' BCM' if self.bcm_only else '', self.agent.backend,
                          len(self.indices), selection,
                          str(self.bandit_result_file or 'session only') if self.selector else 'disabled')

    def _log_status(self, packet):
        now = perf_counter()
        if now - self.last_status_time < 60:
            return
        phase = packet.match_info.match_phase
        score = '-'.join(str(t.score) for t in sorted(packet.teams, key=lambda t: t.team_index))
        timing = ('p50=%.2f p95=%.2f p99=%.2f ms' % tuple(np.percentile(self.inference_ms, [50,95,99]))
                  if self.inference_ms else 'no decisions')
        bandit = (f'; bandit episodes for/against/scoreless={self.selector.results.tolist()}, '
                  f'last={self.selector.last_result}' if self.selector else '')
        self._logger.info('%s%s: phase=%s score=%s cars=%d %s=%s; %d decisions, %s; '
                          'over 33.3ms=%d, missed slots=%d%s',
                          self.variant.name, ' BCM' if self.bcm_only else '',
                          str(phase).removeprefix('MatchPhase.'), score, len(self.indices),
                          'bandit' if self.selector else 'mode', self.agent.mode.value,
                          len(self.inference_ms), timing, self.slow_decisions,
                          self.missed_decisions, bandit)
        self.inference_ms.clear()
        self.missed_decisions = self.slow_decisions = 0
        self.last_status_time = now

    def retire(self):
        if self.selector:
            self.selector.bandit.close()

    def reset(self):
        self.queue = np.zeros((6, len(self.ordered_indices), 8))
        self.obs = RippleObs(self.queue)
        self.parser = RippleAction(self.queue)
        self.obs.reset(self.game_state)
        if self.bcm_only:
            # Preserve the old BCM port's continuation-bias workaround.
            self.queue[:,:,[0,6]] = 1.
        self.actions = np.nan_to_num(self.queue[-1], nan=0.).copy()
        self.clock = DecisionClock()
        self.last_decision = None

    def get_outputs(self, packet):
        phase = packet.match_info.match_phase
        if not packet.balls or not packet.players:
            return {i: ControllerState() for i in self.indices}
        frame = packet.match_info.frame_num
        restart = self.last_packet_frame is not None and frame < self.last_packet_frame
        if restart:
            # Compat has frame-dependent physics history as well as our own queue.
            self.game_state = V1GameState(self.field_info, self.match_config, tick_skip=4)
        self.last_packet_frame = frame
        if self.selector:
            previous_mode = self.agent.mode
            previous_results = int(self.selector.results.sum())
            self.agent.mode = self.selector.update(packet, self.team)
            if self.agent.mode != previous_mode or self.selector.results.sum() != previous_results:
                self._logger.debug('Ripple bandit: mode=%s, last result=%s, episodes=%s',
                                   self.agent.mode.value, self.selector.last_result,
                                   self.selector.results.tolist())
        # Compatibility tracking runs on every packet, including countdown.
        self.game_state.update(packet)
        self.ordered_indices = sorted(self.indices)
        signature = (tuple(p.player_id for p in packet.players), tuple(self.ordered_indices))
        kickoff_reset = (phase in (MatchPhase.Countdown, MatchPhase.Kickoff) and
                         self.previous_phase not in (MatchPhase.Countdown, MatchPhase.Kickoff))
        self.previous_phase = phase
        controlled = [self.game_state.players[i] for i in self.ordered_indices
                      if i < len(self.game_state.players)]
        if len(controlled) != len(self.ordered_indices):
            return {i: ControllerState() for i in self.indices}
        # Explicitly preserve index-to-batch mapping even if indices are unsorted.
        controlled_ids = {p.player_id for p in controlled}
        self.game_state.players = controlled + [p for p in self.game_state.players
                                                if p.player_id not in controlled_ids]
        if signature != self.signature or kickoff_reset or restart:
            self.signature = signature
            self.reset()
        if phase not in (MatchPhase.Active, MatchPhase.Kickoff):
            self.clock = DecisionClock()
            self.last_decision = None
            self._log_status(packet)
            return {i: ControllerState() for i in self.indices}

        periods = self.clock.advance(frame)
        if periods:
            start = perf_counter()
            elapsed = 4 if self.last_decision is None else max(1, frame - self.last_decision)
            # During a gap RLBot holds the last controls. Advance the delayed
            # queue by repeating its newest plan; do not infer on stale packets.
            for _ in range(min(periods - 1, 6)):
                self.queue[1:] = self.queue[:-1].copy()
            self.missed_decisions += periods - 1
            self.obs.pre_step(self.game_state, len(controlled), elapsed_ticks=elapsed)
            self.obs.prev_prev_actions[:len(controlled)] = self.actions
            stochastic = self.stochastic_kickoffs and np.all(self.game_state.ball.position[:2] == 0)
            chosen = self.agent.act(self.obs.current_obs, stochastic=stochastic)
            self.actions = self.parser.parse_actions(chosen, self.game_state)
            self.controls = {index: ControllerState(
                *[float(x) for x in self.actions[row, :5]],
                *[bool(x) for x in self.actions[row, 5:]])
                for row, index in enumerate(self.ordered_indices)}
            self.last_decision = frame
            duration_ms = (perf_counter() - start) * 1000
            self.inference_ms.append(duration_ms)
            self.slow_decisions += duration_ms >= 1000 / 30
        self._log_status(packet)
        return self.controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=list(VARIANTS), default=None)
    parser.add_argument('--backend', choices=['onnx','torch'], default='onnx')
    parser.add_argument('--bcm-only', action='store_true', help='Run this version\'s imitation model without its RL policy')
    parser.add_argument('--seed',type=int,help='Seed action sampling for reproducible comparisons')
    parser.add_argument("--model", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--mode", choices=['bandit', *[m.value for m in ActionMode]],
                        help='Default: bandit for RL policies, stochastic for BCM-only')
    parser.add_argument('--bandit-result-file', type=Path,
                        help='Persist bandit statistics in SQLite (default: session only)')
    parser.add_argument("--disable-stochastic-kickoffs", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Check bundled models and meshes, then exit")
    args = parser.parse_args()
    args.mode = args.mode or ('stochastic' if args.bcm_only else 'bandit')
    if args.bcm_only and args.mode not in ('stochastic', 'deterministic'):
        parser.error('--bcm-only supports stochastic or deterministic selection')
    if args.bandit_result_file and args.mode != 'bandit':
        parser.error('--bandit-result-file requires --mode bandit')
    variant = select_variant(args.variant)
    default_model, default_base = model_paths(variant,args.backend,bcm_only=args.bcm_only)
    model = args.model or default_model
    base = (args.base or default_base) if args.mode in ('bandit','kl','inverse-kl') else None
    if args.mode in ('bandit','kl','inverse-kl') and base is None:
        parser.error('Bandit and KL selection require a separate base model')
    if args.self_test:
        print("Loading models...", flush=True)
        agent = Agent(model, base, BANDIT_MODES[0] if args.mode == 'bandit' else args.mode,
                      backend=args.backend, seed=args.seed,
                      available_modes=BANDIT_MODES if args.mode == 'bandit' else None)
        print("Warming inference...", flush=True)
        agent.warmup(3)
        if args.mode == 'bandit':
            print('Checking bandit and SQLite...', flush=True)
            bandit = ModeBandit(variant.agent_id, seed=args.seed)
            try:
                context = bandit.context(['Ripple self-test'], ['Opponent'])
                mode = bandit.select(context)
                bandit.record(context, mode, 1)
                if bandit.select(context) != mode:
                    raise RuntimeError('Bandit failed to retain its successful mode')
            finally:
                bandit.close()
        print("Loading collision meshes...", flush=True)
        observation_builder = RippleObs(np.zeros((6, 3, 8)))
        observation_builder.arena.step(4)
        label = f'{variant.name} BCM' if args.bcm_only else variant.name
        print(f'{label}: {args.backend} models and RocketSim meshes loaded successfully')
        return
    Ripple(model, base, args.mode, not args.disable_stochastic_kickoffs,
           args.variant,args.backend,args.seed,args.bcm_only,args.bandit_result_file).run(
        wants_match_communications=False, wants_ball_predictions=False)


if __name__ == "__main__":
    main()
