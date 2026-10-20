"""Legacy hierarchical Thompson sampling, updated only between episodes."""
import json
import sqlite3
from pathlib import Path

import numpy as np
from rlbot.flat import MatchPhase

from agent import ActionMode


# Preserve the old port's tie-breaking order, including its perfect-win rule.
BANDIT_MODES = (ActionMode.KL, ActionMode.INVERSE_KL,
                ActionMode.STOCHASTIC, ActionMode.DETERMINISTIC)
LIVE_PHASES = (MatchPhase.Active, MatchPhase.Kickoff)
END_PHASES = (MatchPhase.GoalScored, MatchPhase.Replay, MatchPhase.Ended)


class ModeBandit:
    def __init__(self, profile, result_file=None, seed=None):
        if result_file is not None:
            result_file = Path(result_file).expanduser()
            result_file.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(result_file) if result_file else ':memory:')
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        # Aggregate counters avoid scanning an ever-growing episode history.
        # A separate table leaves databases from the old port untouched.
        self.db.execute('''CREATE TABLE IF NOT EXISTS ripple_mode_counts (
            profile TEXT NOT NULL, teammates TEXT NOT NULL, opponents TEXT NOT NULL,
            own_size INTEGER NOT NULL, other_size INTEGER NOT NULL, mode TEXT NOT NULL,
            wins INTEGER NOT NULL, losses INTEGER NOT NULL, draws INTEGER NOT NULL,
            PRIMARY KEY (profile, teammates, opponents, mode))''')
        self.db.commit()

    @staticmethod
    def context(teammates, opponents):
        return (json.dumps(sorted(teammates), ensure_ascii=False),
                json.dumps(sorted(opponents), ensure_ascii=False),
                len(teammates), len(opponents))

    def record(self, context, mode, outcome):
        with self.db:
            self.db.execute('''INSERT INTO ripple_mode_counts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (profile, teammates, opponents, mode) DO UPDATE SET
                wins = wins + excluded.wins, losses = losses + excluded.losses,
                draws = draws + excluded.draws''',
                (self.profile, *context, ActionMode(mode).value,
                 int(outcome > 0), int(outcome < 0), int(outcome == 0)))

    def statistics(self, context):
        # 0: exact matchup, 1: other matchups of the same sizes, 2: other sizes.
        rows = self.db.execute('''SELECT mode,
            CASE WHEN teammates = ? AND opponents = ? THEN 0
                 WHEN own_size = ? AND other_size = ? THEN 1 ELSE 2 END AS scope,
            SUM(wins), SUM(losses) FROM ripple_mode_counts WHERE profile = ?
            GROUP BY mode, scope''', (*context, self.profile))
        counts = np.zeros((3, 2, len(BANDIT_MODES)))
        for mode, scope, wins, losses in rows:
            if mode in BANDIT_MODES:
                counts[scope, :, BANDIT_MODES.index(mode)] = wins, losses
        return counts

    def posterior(self, context):
        counts = self.statistics(context)
        mean_rest = (counts[2, 0] + .5) / (counts[2].sum(axis=0) + 1)
        mean_mode = (counts[1, 0] + mean_rest) / (counts[1].sum(axis=0) + 1)
        return counts[0, 0] + mean_mode, counts[0, 1] + 1 - mean_mode, counts[0]

    def select(self, context):
        alpha, beta, (wins, losses) = self.posterior(context)
        perfect = (wins > 0) & (losses == 0)
        index = np.argmax(perfect) if perfect.any() else np.argmax(self.rng.beta(alpha, beta))
        return BANDIT_MODES[index]

    def close(self):
        self.db.close()


class EpisodeSelector:
    """Credit each goal once, even if scores arrive after the phase transition."""
    def __init__(self, bandit):
        self.bandit = bandit
        self.roster = None
        self.scores = None
        self.frame = None
        self.phase = None
        self.started = False
        self.waiting = False
        self.mode = BANDIT_MODES[0]
        self.results = np.zeros(3, dtype=int)  # goals for, goals against, scoreless
        self.last_result = 'none'

    def _record(self, outcome):
        self.bandit.record(self.context, self.mode, outcome)
        self.results[0 if outcome > 0 else 1 if outcome < 0 else 2] += 1
        self.last_result = 'scored' if outcome > 0 else 'conceded' if outcome < 0 else 'scoreless'
        self.started = False

    def update(self, packet, team):
        phase = packet.match_info.match_phase
        frame = packet.match_info.frame_num
        scores = tuple(next((t.score for t in packet.teams if t.team_index == i), 0)
                       for i in (team, 1 - team))
        roster = (team, tuple((p.player_id, p.name, p.team) for p in packet.players))
        reset = (roster != self.roster or self.frame is None or frame < self.frame or
                 any(a < b for a, b in zip(scores, self.scores or scores)))
        if reset:
            self.roster = roster
            self.context = self.bandit.context(
                [p.name for p in packet.players if p.team == team],
                [p.name for p in packet.players if p.team != team])
            self.mode = self.bandit.select(self.context)
            self.started = self.waiting = False
            self.scores = scores
        just_scored = self.started and scores != self.scores
        if just_scored:
            delta = (scores[0] - self.scores[0]) - (scores[1] - self.scores[1])
            self._record(int(np.sign(delta)))
            self.waiting = True
        if phase in END_PHASES:
            self.waiting = True
            # Wait through replay for a possibly delayed score update.
            if phase == MatchPhase.Ended and self.started:
                self._record(0)
        elif phase == MatchPhase.Countdown or phase in LIVE_PHASES:
            countdown_boundary = phase == MatchPhase.Countdown and self.started
            next_episode = (phase == MatchPhase.Countdown or
                            phase == MatchPhase.Kickoff and self.phase != MatchPhase.Kickoff or
                            phase == MatchPhase.Active and self.phase in END_PHASES)
            if (self.waiting and next_episode or countdown_boundary) and not just_scored:
                if self.started:
                    self._record(0)
                self.mode = self.bandit.select(self.context)
                self.waiting = False
            if phase in LIVE_PHASES and not self.waiting and not just_scored:
                self.started = True
        self.scores, self.frame, self.phase = scores, frame, phase
        return self.mode
