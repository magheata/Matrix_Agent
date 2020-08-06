import Constants
from Domain.Action import Action
from Domain.EpisodeResult import EpisodeResult


class ExperimentService:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def compute_episodes(self, total_episodes):
        solved_eps = 0
        total_reward = 0
        steps_taken_for_completion = []
        for episode in range(total_episodes):
            actions = []
            env_states = []
            state = self.env.reset_action()
            print('episode {}/{}'.format(episode, total_episodes))
            for time in range(20):
                env_states.append(self.env.s.copy())
                action = self.agent.act(state)
                next_state, reward, done = self.env.step_action(action, len(actions))
                total_reward = total_reward + reward
                self.agent.memorize(state, action, reward, next_state, done)
                state = next_state
                actions.append(action)
                if done:
                    print('         - SOLVED! episode {}/{} steps taken: {}'.format(episode, total_episodes,
                                                                                    len(actions)))
                    steps_taken_for_completion.append(len(actions))
                    solved_eps = solved_eps + 1
                    actions_enum = []
                    for i in actions:
                        actions_enum.append(Action(i))
                    self.agent.update_target_model()
                    break
                if len(self.agent.memory) > Constants.BATCH_SIZE:
                    self.agent.replay(Constants.BATCH_SIZE)
        return solved_eps, steps_taken_for_completion, total_reward

    def run_experiment(self, episodes, iterations):
        episode_results = []
        samples_results = {}
        steps_taken = []
        for sample in range(episodes):
            print("sample", sample)
            total_solved, steps_taken_for_completion, total_reward = self.compute_episodes(iterations)
            steps_taken.append(steps_taken_for_completion)
            episode_results.append(EpisodeResult(iterations, steps_taken_for_completion, total_solved, total_reward))
            print("episodes: {} total_solved: {}".format(iterations, total_solved))
            print("\n\n")
            samples_results[sample] = episode_results
            episode_results = []
        return samples_results
