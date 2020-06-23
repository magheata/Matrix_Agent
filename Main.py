from DQNAgent import DQNAgent

import gym

EPISODES = 20

if __name__ == "__main__":
    env = gym.make("env:MatrixEnv-v0")
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    #print(env.s)
    for e in range(EPISODES):
        state = env.reset()
        for time in range(20):
            # env.render()
            action = agent.act(state)
            next_state, reward, done = env.step_action(action)
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    #print(state)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
