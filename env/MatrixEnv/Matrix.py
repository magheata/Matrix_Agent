import sys
import numpy as np
import gym
from gym.vector.utils import spaces
from scipy.spatial import distance
from six import StringIO

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
        self.action_space = spaces.Discrete(4)
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

    def reset(self):
        ...

    def render(self, mode='human'):
        ...

    def close(self):
        ...

    def encode(self, row, col):
        i = row
        i *= 5
        i += col
        return i

    def step_action(self, action):
        assert self.action_space.contains(action)

        current_position = np.where(self.s == 1)
        row = current_position[0]
        col = current_position[1]
        self.s[row, col] = 0

        if action == 0:
            new_row = min(row + 1, self.max_row)
            position = (new_row, col)
        elif action == 1:
            new_row = max(row - 1, 0)
            position = (new_row, col)
        elif action == 2:
            new_col = min(col + 1, self.max_col)
            position = (row, new_col)
        elif action == 3:
            new_col = max(col - 1, 0)
            position = (row, new_col)

        reward = distance.cityblock(position, self.terminal_state)  # calcular distancia Manhattan

        self.s[position[0], position[1]] = 1

        return self.s, reward, position == self.terminal_state

    def decode(self, i):
        out = []
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
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
