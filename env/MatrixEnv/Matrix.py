import numpy as np
from gym.envs.toy_text import discrete
from scipy.spatial import distance


class Matrix(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.shape = np.zeros(shape=(5, 5))
        self.dest_loc = (4, 3)
        agent_loc = (0, 0)
        num_states = 5 * 5
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 4

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
                        new_row = min(row + 1, max_row)
                        position = (new_row, col)
                        reward = distance.cityblock(position, self.dest_loc)  # calcular distancia Manhattan
                    elif action == 1:
                        new_row = max(row - 1, 0)
                        position = (new_row, col)
                        reward = distance.cityblock(position, self.dest_loc)  # calcular distancia Manhattan
                    elif action == 2:
                        new_col = min(col + 1, max_col)
                        position = (row, new_col)
                        reward = distance.cityblock(position, self.dest_loc)  # calcular distancia Manhattan
                    elif action == 3:
                        new_col = max(col - 1, 0)
                        position = (row, new_col)
                        reward = distance.cityblock(position, self.dest_loc)  # calcular distancia Manhattan

                    if agent_loc == self.dest_loc:
                        done = True

                    new_state = self.encode(new_row, new_col)
                    P[state][action].append((1.0, new_state, reward, done))

        super(Matrix, self).__init__(num_states, num_actions, P, initial_state_distrib)

    def step(self, action):
        ...

    def reset(self):
        ...

    def render(self, mode='human'):
        ...

    def close(self):
        ...

    def encode(self, row, col):
        # (5) 5, 5, 4
        i = row
        i *= 5
        i += col
        return i

    def decode(self, i):
        out = []
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)
