from Action import Action
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

    for e in range(EPISODES):
        actions = []
        env_states = []
        state = env.reset_action()
        for time in range(20):
            # env.render()
            env_states.append(env.s.copy())
            action = agent.act(state)
            next_state, reward, done = env.step_action(action, len(actions))
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            actions.append(action)

            if done:
                actions_enum = []
                for i in actions:
                    actions_enum.append(Action(i))
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                print("Total actions: ", len(actions_enum), actions_enum)
                print(*env_states, sep=", \n\n")
                print("\n", env.s)
                print("___________________________________________________\n\n")

                agent.update_target_model()
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
