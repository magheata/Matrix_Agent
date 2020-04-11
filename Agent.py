import gym

class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':

    env = gym.make("env:MatrixEnv-v0")

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 999
    done = False

    movements = [0, 1, 2, 3]

    agent_pos = (0, 4)

    length = len(movements)
    best_state = 0
    best_reward = 99999
    while not done:
        for mov_idx in range(length):
            result = env.step(movements[mov_idx])
            new_state = result[0][0]
            new_position = env.decode(new_state)
            if (result[0][1] < best_reward):
                best_reward = result[0][1]
                best_state = result[0][0]
        done = result[0][2]
        env.s = best_state
        new_position = env.decode(env.s)
        print(list(new_position))

    env.close()
