import gym

from Agent import Agent

if __name__ == '__main__':

    env = gym.make("env:MatrixEnv-v0")
    env.seed(0)

    agent = Agent(env.action_space)

    episode_count = 100
    reward = 999
    done = False

    agent_pos = (0, 0)

    best_state = 0
    best_reward = 99999

    state = env.reset()
    best_action = 0

    env.s = env.encode(agent_pos[0], agent_pos[1])
    env.render()
    print("[%d, %d]" % (agent_pos[0], agent_pos[1]))
    while not done:
        initial_state = env.s
        for action_idx in range(len(agent.movements)):
            env.s = initial_state
            next_state, reward, doneRr = env.step_action(agent.movements[action_idx])
            if reward < best_reward:
                best_action = action_idx
                best_reward = reward
                best_state = next_state
        env.s = initial_state
        current_state, reward, done = env.step_action(agent.movements[best_action])
        if done:
            new_position = env.decode(env.s)
            env.render()
            print(list(new_position))
            break
        new_position = env.decode(env.s)
        env.render()
        print(list(new_position))

    env.close()
