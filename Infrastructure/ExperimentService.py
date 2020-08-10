import Constants
import numpy as np
from Domain.Action import Action
from Domain.EpisodeResult import EpisodeResult

from random import seed
from random import randint

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

    def run_experiment_eps(self, episodes, iterations):
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

    def run_experiment_change_dim(self, episodes, iterations):
        changeDim = False
        episode_results = []
        samples_results = {}
        steps_taken = []
        for sample in range(episodes):
            if np.random.rand() <= 0.5:
                changeDim = True
            if changeDim:
                terminal_state = self.env.terminal_state
                start_state = self.env.start_state
                print(max(terminal_state))
                print(max(start_state))
                newDimension = randint(max(max(terminal_state), max(start_state)), 30)
                print("old dimension: {} new dimension: {}".format(self.env.dimension, newDimension))
                self.env.changeDimension(newDimension)
                self.agent.setStateSize(self.env.observation_space.n)
            print("sample", sample)
            total_solved, steps_taken_for_completion, total_reward = self.compute_episodes(iterations)
            steps_taken.append(steps_taken_for_completion)
            episode_results.append(EpisodeResult(iterations, steps_taken_for_completion, total_solved, total_reward))
            print("episodes: {} total_solved: {}".format(iterations, total_solved))
            print("\n\n")
            samples_results[sample] = episode_results
            episode_results = []
            changeDim = False
        return samples_results

    def run_experiment_change_goal(self, episodes, iterations):
        changeGoal = False
        episode_results = []
        samples_results = {}
        steps_taken = []
        for sample in range(episodes):
            if np.random.rand() <= 0.5:
                changeGoal = True
            if changeGoal:
                old_distance = self.env.distance_from_start_to_goal
                terminal_state = self.env.terminal_state
                start_state = self.env.start_state
                col = randint(0, self.env.dimension - 1)
                row = randint(0, self.env.dimension - 1)
                while ((row, col) == terminal_state) or ((row, col) == start_state):
                    col = randint(0, self.env.dimension - 1)
                    row = randint(0, self.env.dimension - 1)
                new_terminal_state = (row, col)
                print("old terminal state: {} new terminal state: {}".format(terminal_state, new_terminal_state))
                self.env.changeTerminalState(new_terminal_state)
                print("Old distance: {} New distance: {}".format(old_distance, self.env.distance_from_start_to_goal))

            print("sample", sample)
            total_solved, steps_taken_for_completion, total_reward = self.compute_episodes(iterations)
            steps_taken.append(steps_taken_for_completion)
            episode_results.append(EpisodeResult(iterations, steps_taken_for_completion, total_solved, total_reward))
            print("episodes: {} total_solved: {}".format(iterations, total_solved))
            print("\n\n")
            samples_results[sample] = episode_results
            episode_results = []
            changeGoal = False
        return samples_results

    def run_experiment_change_origin(self, episodes, iterations):
        changeOrigin = False
        episode_results = []
        samples_results = {}
        steps_taken = []
        for sample in range(episodes):
            if np.random.rand() <= 0.5:
                changeOrigin = True
            if changeOrigin:
                old_distance = self.env.distance_from_start_to_goal
                terminal_state = self.env.terminal_state
                start_state = self.env.start_state
                col = randint(0, self.env.dimension - 1)
                row = randint(0, self.env.dimension - 1)
                while ((row, col) == terminal_state) or ((row, col) == start_state):
                    col = randint(0, self.env.dimension - 1)
                    row = randint(0, self.env.dimension - 1)
                new_start_state = (row, col)
                print("old origin state: {} new origin state: {}".format(start_state, new_start_state))
                self.env.changeStartState(new_start_state)
                print("Old distance: {} New distance: {}".format(old_distance, self.env.distance_from_start_to_goal))
            print("sample", sample)
            total_solved, steps_taken_for_completion, total_reward = self.compute_episodes(iterations)
            steps_taken.append(steps_taken_for_completion)
            episode_results.append(EpisodeResult(iterations, steps_taken_for_completion, total_solved, total_reward))
            print("episodes: {} total_solved: {}".format(iterations, total_solved))
            print("\n\n")
            samples_results[sample] = episode_results
            episode_results = []
            changeOrigin = False
        return samples_results

    def run_experiment_disable_tile(self, episodes, iterations):
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