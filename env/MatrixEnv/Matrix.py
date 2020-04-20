import sys
import numpy as np
import gym
from scipy.spatial import distance
from six import StringIO
from gym.envs.toy_text import discrete

MAP = [
    "+---------+",
    "|_|_|_|_|_|",
    "|_|_|_|_|_|",
    "|_|_|_|_|_|",
    "|_|_|_|_|_|",
    "|_|_|_|_|_|",
    "+---------+",
]

class Matrix(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self._kwargs = {} if kwargs is None else kwargs

        self.desc = np.asarray(MAP, dtype='c')
        self.dimension = kwargs.get('dimension')
        num_rows = self.dimension
        num_columns = self.dimension
        self.shape = np.zeros(shape=(num_rows, num_columns))
        self.dest_loc = kwargs.get('goal')
        agent_loc = (0, 0)
        num_states = self.dimension * self.dimension

        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = kwargs.get('actions')

        P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                state = self.encode(row, col)
                if agent_loc != self.dest_loc:
                    initial_state_distrib[state] += 1
                for action in range(num_actions):
                    # defaults
                    new_row, new_col, new_agent_loc = row, col, agent_loc
                    done = False
                    agent_loc = (row, col)

                    if action == 0:
                        new_row = min(row + 1, self.max_row)
                    elif action == 1:
                        new_row = max(row - 1, 0)
                    elif action == 2:
                        new_col = min(col + 1, self.max_col)
                    elif action == 3:
                        new_col = max(col - 1, 0)
                    if agent_loc == self.dest_loc:
                        done = True
                    new_state = self.encode(new_row, new_col)
                    P[state][action].append((1.0, new_state, done))

        super(Matrix, self).__init__(num_states, num_actions, P, initial_state_distrib)

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
        current_position = list(self.decode(self.s))
        row = current_position[0]
        col = current_position[1]

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

        reward = distance.cityblock(position, self.dest_loc)  # calcular distancia Manhattan

        self.s = self.encode(position[0], position[1])

        return self.s, reward, position == self.dest_loc

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
