import gym


class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.movements = [0, 1, 2, 3]

    def act(self, state, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':

    env = gym.make("env:MatrixEnv-v0")

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.seed(0)
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 999
    done = False

    agent_pos = (0, 4)

    best_state = 0
    best_reward = 99999
    state = env.reset()
    best_action = 0
    env.s = env.encode(0, 0)
    env.render()
    while not done:
        initial_state = env.s
        for action_idx in range(len(agent.movements)):
            env.s = initial_state
            next_state, reward, done, _ = env.step(agent.movements[action_idx])
            if reward < best_reward:
                best_action = action_idx
                best_reward = reward
                best_state = next_state
        env.s = initial_state
        current_state, reward, done, _ = env.step(agent.movements[best_action])
        if done:
            break
        new_position = env.decode(env.s)
        env.render()
        print(list(new_position))

    env.close()
