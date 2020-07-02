import sys
import numpy as np
import gym
import Constants

from gym.vector.utils import spaces
from scipy.spatial import distance
from six import StringIO

from Action import Action

MAP = [
    "+---------+",
    "|_|_|_|_|_|",
    "|_|_|_|_|_|",
    "|_|_|_|_|_|",
    "|_|_|_|_|_|",
    "|_|_|_|_|_|",
    "+---------+",
]


class Matrix(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.desc = np.asarray(MAP, dtype='c')
        self.dimension = kwargs.get('dimension')
        self.n_states = self.dimension ** 2
        self.action_space = spaces.Discrete(len(list(map(int, Action))))
        self.observation_space = spaces.Discrete(self.n_states)  # with absorbing state
        self.num_rows = self.dimension
        self.num_columns = self.dimension
        self.shape = np.zeros(shape=(self.num_rows, self.num_columns))
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.terminal_state = kwargs.get('goal')
        self.start_state = kwargs.get('start_state')
        self.s = np.zeros((self.dimension, self.dimension))
        self.s[self.start_state[0], self.start_state[1]] = 1
        self.s[self.terminal_state[0], self.terminal_state[1]] = 2
        self.initial_state = self.s.copy()
        self.distance_from_start_to_goal = distance.cityblock(self.start_state, self.terminal_state)

    def reset_action(self):
        self.s = self.initial_state.copy()
        return self.s

    def render(self, mode='human'):
        ...

    def close(self):
        ...

    def encode(self, row, col):
        i = row
        i *= self.dimension
        i += col
        return i

    def get_next_position(self, action, row, col):
        parsed_action = Action(action)
        use_penalty = False

        if parsed_action == Action.DOWN:
            new_row = min(row + 1, self.max_row)
            if row + 1 > self.max_row:
                use_penalty = True
            position = (new_row, col)
        elif parsed_action == Action.UP:
            new_row = max(row - 1, 0)
            if row - 1 < 0:
                use_penalty = True
            position = (new_row, col)
        elif parsed_action == Action.RIGHT:
            new_col = min(col + 1, self.max_col)
            if col + 1 > self.max_col:
                use_penalty = True
            position = (row, new_col)
        elif parsed_action == Action.LEFT:
            new_col = max(col - 1, 0)
            if col - 1 < 0:
                use_penalty = True
            position = (row, new_col)
        return position, use_penalty

    def step_action(self, action, steps_taken):
        assert self.action_space.contains(action)
        row, col = self.get_pos_components()
        self.s[row, col] = 0
        position, use_penalty = self.get_next_position(action, row, col)
        self.s[position[0], position[1]] = 1
        return self.s, self.determine_reward(position, use_penalty, steps_taken), position == self.terminal_state

    def determine_reward(self, position, use_penalty, steps_taken):
        reward = distance.cityblock(position, self.terminal_state)  # calcular distancia Manhattan

        if use_penalty:
            reward = reward + Constants.PENALTY_OUT_OF_RANGE

        #mirar max acciones

        if steps_taken < self.distance_from_start_to_goal:
            reward = reward + Constants.REWARD_WITHIN_DISTANCE

        return reward

    def get_pos_components(self):
        current_position = np.where(self.s == 1)
        return current_position[0], current_position[1]

    def decode(self, i):
        out = [i % self.dimension]
        i = i // self.dimension
        out.append(i)
        assert 0 <= i < self.dimension
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col = self.decode(self.s)

        def ul(x): return "_" if x == " " else x

        out[1 + taxi_row][2 * taxi_col + 1] = gym.utils.colorize(
            ul(out[1 + taxi_row][2 * taxi_col + 1]), 'blue', highlight=True)

        di, dj = self.dest_loc
        out[1 + di][2 * dj + 1] = gym.utils.colorize(out[1 + di][2 * dj + 1], 'magenta', highlight=True)
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

    def step(self, action):
        pass

    def reset(self):
        pass
