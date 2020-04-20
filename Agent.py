import gym


class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.movements = [0, 1, 2, 3]

    def act(self, state, reward, done):
        return self.action_space.sample()
