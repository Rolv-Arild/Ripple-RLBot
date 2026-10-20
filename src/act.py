from typing import Any

import numpy as np
from rlgym_compat import V1GameState


class RippleLookupAction:
    def __init__(self):
        super().__init__()
        self.lookup_table = self.make_lookup_table([-1, 0, 0.5, 1], 5, 5, 16)

    @staticmethod
    def make_lookup_table(throttle_bins: Any = 3,
                          steer_bins: Any = 3,
                          torque_bins: Any = 3,
                          flip_bins: Any = 8,
                          include_stall=False) -> np.ndarray:
        # Parse bins
        def parse_bin(b, endpoint=True):
            if isinstance(b, int):
                b = np.linspace(-1, 1, b, endpoint=endpoint)
            else:
                b = np.array(b)
            return b

        throttle_bins = parse_bin(throttle_bins)
        steer_bins = parse_bin(steer_bins)
        torque_bins = parse_bin(torque_bins)
        flip_bins = (parse_bin(flip_bins, endpoint=False) + 1) * np.pi  # Split a circle into equal segments in [0, 2pi)

        actions = []

        # Ground
        pitch = roll = jump = 0
        for throttle in throttle_bins:
            for steer in steer_bins:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        yaw = steer
                        actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # Aerial
        jump = handbrake = 0
        for pitch in torque_bins:
            for yaw in torque_bins:
                for roll in torque_bins:
                    if pitch == roll == 0 and np.isclose(yaw, steer_bins).any():
                        continue  # Duplicate with ground
                    magnitude = max(abs(pitch), abs(yaw), abs(roll))
                    if magnitude < 1:
                        continue  # Duplicate rotation direction, only keep max magnitude
                    for boost in (0, 1):
                        throttle = boost
                        steer = yaw
                        actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # Flips and jumps
        jump = handbrake = 1  # Enable handbrake for potential wavedashes
        yaw = steer = 0  # Only need roll for sideflip
        angles = [np.nan] + [v for v in flip_bins]
        for angle in angles:
            if np.isnan(angle):
                pitch = roll = 0  # Empty jump
            else:
                pitch = np.sin(angle)
                roll = np.cos(angle)
                # Project to square of diameter 2 because why not
                magnitude = max(abs(pitch), abs(roll))
                pitch /= magnitude
                roll /= magnitude
            for boost in (0, 1):
                throttle = boost
                actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
        if include_stall:
            actions.append([0, 0, 0, 1, -1, 1, 0, 1])  # One final action for stalling

        actions = np.array(actions)
        actions = np.round(actions, 3)  # Convert to numpy and remove floating point errors
        assert len(np.unique(actions, axis=0)) == len(actions), 'Duplicate actions found'

        return actions

    def parse_actions(self, actions: Any, state: V1GameState) -> np.ndarray:
        return self.lookup_table[actions.squeeze(-1) if actions.ndim > 1 else actions]


class DelayedAction:
    def __init__(self, parser, action_queue: np.ndarray, end_index: int = -1):
        """
        DelayedAction maintains a queue of actions to execute and adds parsed actions to the queue.

        :param parser: the action parser to parse actions that are then added to the queue.
        :param action_queue: Shared with the observation builder, shape
                             (6, controlled_players, 8), newest action first.
                             NaNs mark a player's queue as unknown; their first
                             selection fills that entire column after a reset.
        """
        super().__init__()
        # assert action_queue.ndim == 3 and action_queue.shape[1:] == (6, 8)
        self.parser = parser
        self.action_queue = action_queue
        self.end_index = end_index

    def parse_actions(self, actions: Any, state: V1GameState) -> np.ndarray:
        new_actions = self.parser.parse_actions(actions, state)
        p = slice(0, new_actions.shape[0])  # Select players
        invalid = np.isnan(self.action_queue[:, p])
        if invalid.any():
            invalid_players = np.where(invalid.any(axis=(0, 2)))[0]
            for i in invalid_players:
                self.action_queue[:, i] = new_actions[i]
        next_actions = self.action_queue[self.end_index][p].copy()
        self.action_queue[1:, p] = self.action_queue[:-1, p]  # Shift queue
        self.action_queue[0, p] = new_actions
        return next_actions


def RippleAction(action_queue):
    return DelayedAction(RippleLookupAction(), action_queue)
