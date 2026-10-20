import os
import random
from pathlib import Path
from typing import Any

import RocketSim as rs
import numpy as np
from rlgym_compat import ORANGE_TEAM
from rlgym_compat import V1PlayerData, V1GameState

MESH_DIRECTORY = Path(__file__).resolve().parent / "collision_meshes"
if not (MESH_DIRECTORY / "soccar/mesh_0.cmf").is_file():
    raise FileNotFoundError(f"Missing RocketSim collision meshes: {MESH_DIRECTORY}")
os.environ["RS_COLLISION_MESHES"] = str(MESH_DIRECTORY)
# Use an explicit asset path instead of depending on the working directory.
try:
    rs.init(str(MESH_DIRECTORY))
except RuntimeError as error:
    if str(error) != "Already inited":
        raise

BOOST_LOCATIONS = np.array((
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (- 940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (- 940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
)
)

TICK_SKIP = 4
TICKS_PER_SECOND = 120

INVERT_TEAM = np.array([-1, -1, 1] * 5 + [-1] + [1] * 16)
INVERT_SIDE = np.array([-1, 1, 1] * 2 + [1, -1, -1] + [-1, 1, 1] * 2 + [1] * 17)
INVERT_SIDE_RELATIVE = np.array([1, -1, 1] * 2 + [-1, 1, -1] + [1, -1, 1] * 2 + [1] * 12 + [-1] + [1] * 4)
INVERT_SIDE_ACTIONS = np.array([1, -1, 1, -1, -1, 1, 1, 1])
NORM = np.array([2300] * 6 + [6] * 3 + [1] * 6 + [1, 3, 1, 1, 1, 1, 12, 24, 150, 120, 1, 1, 1, 1, 1, 1, 1])
BOOST_LOCATIONS = np.array(BOOST_LOCATIONS)

BALL_PRED_TOLERANCE = np.array([1.] * 6 + [0.1] * 3)  # Training tolerance.
DEFAULT_DODGE_DEADZONE = 0.5
GOAL_THRESHOLD = 5215.5  # Tested in-game with BakkesMod


class RippleObs:
    def __init__(self, action_queue: np.ndarray, tick_skip=TICK_SKIP, ball_steps=5, ball_step_size_seconds=1,
                 scoreboard=None, infinite_boost_ratio=0, env=None):
        super().__init__()
        self.action_queue = action_queue
        self.tick_skip = tick_skip

        self.ball_steps = ball_steps
        self.ball_step_size_ticks = round(ball_step_size_seconds * TICKS_PER_SECOND) // self.tick_skip

        # self.past_actions = past_actions
        # self.future_actions = future_actions

        self.demo_timers = None
        self.boost_timers = None
        self.ball_predictions = np.zeros((1 + ball_steps * self.ball_step_size_ticks, 9))

        self.current_obs = None

        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        # self.ball = Ball(BALL)

        self.n = 0  # Counter for player to return obs for

        self.scoreboard = scoreboard
        self.infinite_boost_ratio = infinite_boost_ratio
        self.env = env
        self.boost_consumption = 1

        self.ball_pred_resets = 0
        self.steps = 0
        self.resets = 0

        # Arrays for player timers
        self.boost_times = None
        self.prev_on_grounds = None
        self.is_jumpings = None
        self.has_jumpeds = None
        self.jump_times = None
        self.air_times = None
        self.has_double_jumpeds = None
        self.has_flippeds = None
        self.flip_times = None
        self.flip_dirs = None
        self.handbrakes = None
        self.prev_prev_actions = None
        self.extra_shared_data_dash_time_cars = None

    def reset(self, initial_state: V1GameState):
        self.resets += 1
        num_players = len(initial_state.players)
        self.demo_timers = np.zeros(num_players)
        self.boost_timers = np.zeros(len(initial_state.boost_pads))
        # self.reset_ball_slices(initial_state)
        if initial_state.ball.position[0] == 0 and initial_state.ball.position[1] == 0 \
                and all(np.all(abs(p.car_data.linear_velocity) < 10) for p in initial_state.players):
            self.action_queue[:] = 0  # Kickoffs start with all no-op
        else:
            self.action_queue[:] = np.nan  # Indicates reset to action parser

        if self.env is not None:
            if self.scoreboard is not None and self.scoreboard.random_resets:
                inf_boost = random.random() < self.infinite_boost_ratio
                self.boost_consumption = 0 if inf_boost else 1
            else:
                self.boost_consumption = 1
            self.env.update_settings(boost_consumption=self.boost_consumption)

        # Arrays for player timers
        self.boost_times = np.zeros(num_players)
        self.prev_on_grounds = np.zeros(num_players, dtype=bool)
        self.is_jumpings = np.zeros(num_players, dtype=bool)
        self.has_jumpeds = np.zeros(num_players, dtype=bool)
        self.jump_times = np.zeros(num_players)
        self.air_times = np.zeros(num_players)
        self.has_double_jumpeds = np.zeros(num_players, dtype=bool)
        self.has_flippeds = np.zeros(num_players, dtype=bool)
        self.flip_times = np.zeros(num_players)
        self.flip_dirs = np.zeros((num_players, 2))
        self.handbrakes = np.zeros(num_players, dtype=bool)
        self.prev_prev_actions = np.zeros((num_players, 8))
        self.extra_shared_data_dash_time_cars = np.zeros(num_players)

    def pre_step(self, state: V1GameState, controlled_count=None, elapsed_ticks=4) -> bool:
        if self.scoreboard is not None and self.scoreboard.random_resets and self.env is not None:
            boost_consumption_rate = self.boost_consumption
            if boost_consumption_rate <= 0:
                for player in state.players:
                    player.boost_amount = float("inf")
            else:
                for player in state.players:
                    player.boost_amount /= boost_consumption_rate

        did_reset = False

        n_players = len(state.players)
        n_views = n_players if controlled_count is None else controlled_count
        n_slots = max(6, n_players)  # Training uses five other-player slots, masked by team=0.
        if n_players != len(self.demo_timers):
            print(f"WARNING: Number of players changed from {len(self.demo_timers)} to {n_players}, "
                  f"resetting obs builder")
            self.reset(state)
            did_reset = True

        # Players, (pos|vel|ang_vel|forward|up)_[xyz] + team
        # + demo_timer + on_ground + has_(jump|flip) + boost_amount
        # + (boost|jump|air|flip|handbrake)_timer + flip_dir_[xy]
        # + is_reset + is_flipped_kickoff + is_flipped + is_swapped
        players = np.zeros((n_views, n_slots, 32))
        # Ball, (pos|vel|ang_vel)_[xyz] for each time slice
        balls = np.zeros((n_views, 1 + self.ball_steps, 9))
        # Action queue, t to t+6 (past actions for physics info, future actions for reaction time)
        actions = np.zeros((n_views, self.action_queue.shape[0], 8))
        # Boosts, pos_[xyz] + boost_amount + demo_timer for each boost pad
        boosts = np.zeros((n_views, len(self.boost_timers), 5))

        if elapsed_ticks == self.tick_skip:
            self.update_ball_slices(state)
        else:
            # Cached slices are spaced four ticks apart. A late packet may have a
            # different phase, so regenerate instead of silently shifting once.
            self.reset_ball_slices(state)
            self.ball_predictions[0] = np.concatenate((state.ball.position,
                state.ball.linear_velocity, state.ball.angular_velocity))
        nominal_skip = self.tick_skip
        self.tick_skip = elapsed_ticks
        try:
            self.update_timers(state)
        finally:
            self.tick_skip = nominal_skip

        balls[:] = self.ball_predictions[::self.ball_step_size_ticks]

        is_kickoff = state.ball.position[0] == 0 and state.ball.position[1] == 0

        # Populate players array
        for i, player in enumerate(state.players):
            players[:, i, 0:3] = player.car_data.position
            players[:, i, 3:6] = player.car_data.linear_velocity
            players[:, i, 6:9] = player.car_data.angular_velocity
            players[:, i, 9:12] = player.car_data.forward()
            players[:, i, 12:15] = player.car_data.up()
            players[:, i, 15] = 2 * player.team_num - 1
            if player.is_demoed and self.demo_timers[i] <= 0:
                self.demo_timers[i] = 3
            players[:, i, 16] = self.demo_timers[i]
            players[:, i, 17] = player.on_ground
            players[:, i, 18] = player.has_jump
            players[:, i, 19] = player.has_flip
            players[:, i, 20] = min(player.boost_amount, 2)
            players[:, i, 21] = self.boost_times[i]
            players[:, i, 22] = self.jump_times[i]
            players[:, i, 23] = self.air_times[i]
            players[:, i, 24] = self.flip_times[i]
            players[:, i, 25] = self.handbrakes[i]
            players[:, i, 26:28] = self.flip_dirs[i]
            is_reset = np.isnan(self.action_queue[:, i, :]).any() if i < self.action_queue.shape[1] else False
            players[:, i, 28] = is_reset  # Indicates next action will fill action queue
            # Will be set later:
            players[:, i, 29] = is_kickoff  # is_flipped (kickoff only), will be flipped later
            # players[:, i, 30] = 0  # is_flipped
            # players[:, i, 31] = 0  # is_swapped
            if not is_reset and i < n_views and i < self.action_queue.shape[1]:
                actions[i] = self.action_queue[:, i, :]

        boosts[:, :, :3] = BOOST_LOCATIONS
        boosts[:, :, 3] = 0.12 + 0.88 * (boosts[:, :, 2] > 72)
        boosts[:, :, 4] = self.boost_timers / 10

        # Set obs to be relative, blue-side-only and left-only
        for i, player in enumerate(state.players[:n_views]):
            rot = player.car_data.rotation_mtx()

            players[i, [0, i]] = players[i, [i, 0]]  # Current player always at index 0

            players[i, 1:, 0:6] -= players[i, 0, 0:6]
            players[i, 1:, 0:15] = (players[i, 1:, 0:15].reshape(-1, 3) @ rot).reshape(-1, 15)

            balls[i, :, 0:6] -= players[i, 0, 0:6]
            balls[i, :, 0:9] = (balls[i, :, 0:9].reshape(-1, 3) @ rot).reshape(-1, 9)

            boosts[i, :, 0:3] -= players[i, 0, 0:3]
            boosts[i, :, 0:3] = boosts[i, :, 0:3] @ rot

            if player.team_num == ORANGE_TEAM:
                # balls[i] *= INVERT_TEAM[:9]
                players[i, 0] *= INVERT_TEAM
                players[i, 1:, 15:] *= INVERT_TEAM[15:]  # Leave relative coords unchanged
                # boosts[i, :, :3] *= INVERT_TEAM[:3]
                players[i, 0, 31] = 1  # is_swapped (not used in network, just for data checking)

            if players[i, 0, 0] < 0:
                balls[i] *= INVERT_SIDE_RELATIVE[:9]
                players[i, 0] *= INVERT_SIDE
                players[i, 1:, :] *= INVERT_SIDE_RELATIVE
                boosts[i, :, :3] *= INVERT_SIDE_RELATIVE[:3]
                actions[i] *= INVERT_SIDE_ACTIONS
                players[i, 0, 29] *= -1  # Indicates side only during kickoff
                players[i, 0, 30] = 1  # is_flipped (need it to adjust action at the end)

        # Normalize everything
        balls[..., :9] /= NORM[:9]
        players /= NORM
        boosts[..., :3] /= NORM[:3]

        main_players = players[:, 0]
        other_players = players[:, 1:, :21]  # Leave out timers and is_flipped/is_swapped

        self.current_obs = tuple(np.ascontiguousarray(x, dtype=np.float32) for x in
                                 (main_players, other_players, balls, actions, boosts))
        self.n = 0  # Counter for player to return obs for

        self.steps += 1

        return did_reset

    def build_obs(self, player: V1PlayerData, state: V1GameState, previous_action: np.ndarray) -> Any:
        main_players, other_players, balls, actions, boosts = self.current_obs
        self.prev_prev_actions[self.n] = previous_action
        idx = [self.n]
        self.n += 1
        return main_players[idx], other_players[idx], balls[idx], actions[idx], boosts[idx]

    @staticmethod
    def ball_to_array(ball):
        state = ball.get_state()
        return np.array([
            state.pos.x,
            state.pos.y,
            state.pos.z,
            state.vel.x,
            state.vel.y,
            state.vel.z,
            state.ang_vel.x,
            state.ang_vel.y,
            state.ang_vel.z,
        ])

    def update_ball_slices(self, state: V1GameState):
        slice_ball = self.ball_predictions[1]

        state_ball = state.ball
        state_ball = np.concatenate([
            state_ball.position,
            state_ball.linear_velocity,
            state_ball.angular_velocity
        ])

        if (abs(slice_ball - state_ball) < BALL_PRED_TOLERANCE).all():  # Reuse predictions if possible
            self.add_ball_slices()
        else:
            self.reset_ball_slices(state)  # TODO make sure this works and is not used often
        self.ball_predictions[0] = state_ball  # Just to make current position 100% accurate

    def add_ball_slices(self):
        self.ball_predictions[:-1] = self.ball_predictions[1:]
        b = self.ball_predictions[-1]
        if abs(b[1]) >= GOAL_THRESHOLD:
            self.ball_predictions[-1] = b
        else:
            self.arena.step(self.tick_skip)
            self.ball_predictions[-1] = self.ball_to_array(self.arena.ball)

    def reset_ball_slices(self, state: V1GameState):
        self.ball_pred_resets += 1
        self.arena.ball.set_state(rs.BallState(pos=rs.Vec(*state.ball.position),
                                               vel=rs.Vec(*state.ball.linear_velocity),
                                               ang_vel=rs.Vec(*state.ball.angular_velocity)))
        for n in range(1, len(self.ball_predictions)):
            self.arena.step(self.tick_skip)
            b = self.ball_to_array(self.arena.ball)
            if abs(b[1]) >= GOAL_THRESHOLD:
                self.ball_predictions[n:] = b
                break
            self.ball_predictions[n] = b

    def update_timers(self, state: V1GameState):
        is_available = state.boost_pads == 1
        self.boost_timers -= self.tick_skip / TICKS_PER_SECOND
        self.boost_timers[is_available] = 0
        new_captures = (~is_available) & (self.boost_timers <= 0)
        self.boost_timers[new_captures] = 6 * (BOOST_LOCATIONS[new_captures, 2] > 72) + 4
        self.boost_timers = np.clip(self.boost_timers, 0, 10)

        self.demo_timers -= self.tick_skip / TICKS_PER_SECOND
        self.demo_timers[self.demo_timers < 0] = 0

        self.update_player_timers(state)

    def update_player_timers(self, state):
        for i, player in enumerate(state.players):
            if i < self.action_queue.shape[1]:
                # _, _, _, _, last_actions = self.get_mirror_invert(player, i, state)
                last_actions = np.nan_to_num(self.action_queue[-1, i], nan=0.)
            else:
                last_actions = np.zeros(8)

            # Boost time updates
            if not last_actions[6] and self.boost_times[i] == 12.:
                self.boost_times[i] = 0.
            elif self.boost_times[i] != 0. or last_actions[6]:
                self.boost_times[i] += self.tick_skip  # self.time_interval * 120.
                self.boost_times[i] = min(self.boost_times[i], 12.)
            else:
                self.boost_times[i] = 0.

            # Jump time updates
            if self.prev_on_grounds[i] and not self.is_jumpings[i]:
                self.has_jumpeds[i] = False

            if self.is_jumpings[i]:
                self.is_jumpings[i] = self.jump_times[i] < 3. or \
                                      (last_actions[5] and self.jump_times[i] < 24.)
            elif last_actions[5] and not self.prev_prev_actions[i][5] and \
                    self.prev_on_grounds[i]:
                self.is_jumpings[i] = True
                self.jump_times[i] = 0.

            if self.is_jumpings[i]:
                self.has_jumpeds[i] = True
                self.jump_times[i] += self.tick_skip  # self.time_interval * 120.
                self.jump_times[i] = min(self.jump_times[i], 24.)
            else:
                self.jump_times[i] = 0.

            # Dash time updates
            if player.on_ground and self.has_flippeds[i] and self.flip_times[i] < 120.:
                self.extra_shared_data_dash_time_cars[i] = self.flip_times[i]
            else:
                self.extra_shared_data_dash_time_cars[i] = -1.

            # Airtime and fliptime updates
            if player.on_ground:
                self.has_double_jumpeds[i] = False
                self.has_flippeds[i] = False
                self.air_times[i] = 0.
                self.flip_times[i] = 0.
                self.flip_dirs[i, :] = 0.
            else:
                if self.has_jumpeds[i] and not self.is_jumpings[i]:
                    self.air_times[i] += self.tick_skip  # self.time_interval * 120.
                    self.air_times[i] = min(self.air_times[i], 150.)
                else:
                    self.air_times[i] = 0.

                if self.has_jumpeds[i] and last_actions[5] and \
                        not self.prev_prev_actions[i][5] and self.air_times[i] < 150. and \
                        not self.has_double_jumpeds[i] and not self.has_flippeds[i]:
                    should_flip = np.sum(np.abs(last_actions[[2, 3, 4]])) >= DEFAULT_DODGE_DEADZONE
                    if should_flip:
                        self.flip_times[i] = 0.
                        self.has_flippeds[i] = True
                        flipdir = np.array([-last_actions[2], last_actions[3] + last_actions[4]])
                        if flipdir[0] != 0. or flipdir[1] != 0.:
                            flipdir /= np.linalg.norm(flipdir)
                        self.flip_dirs[i] = flipdir
                    else:
                        self.has_double_jumpeds[i] = True

            if self.has_flippeds[i]:
                self.flip_times[i] += self.tick_skip  # self.time_interval * 120.
                self.flip_times[i] = min(self.flip_times[i], 120.)

            # Handbrake updates
            if last_actions[7]:
                self.handbrakes[i] += 5. * self.tick_skip / TICKS_PER_SECOND  # self.time_interval
                self.handbrakes[i] = min(self.handbrakes[i], 1.)
            else:
                self.handbrakes[i] -= 2. * self.tick_skip / TICKS_PER_SECOND  # self.time_interval
                self.handbrakes[i] = max(self.handbrakes[i], 0.)
            self.prev_on_grounds[i] = player.on_ground
