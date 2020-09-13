import Constants
import numpy as np
from Domain.Action import Action
from Domain.EpisodeResult import EpisodeResult

from random import randint


class ExperimentService:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def compute_iteration(self, episodes, steps):
        solved_eps = 0
        steps_taken_for_completion = []
        episode_rewards = []
        for episode in range(episodes):
            actions = []
            env_states = []
            episode_reward = []
            state = self.env.reset_action()
            print('episode {}/{}'.format(episode, episodes))
            for step in range(steps):
                env_states.append(self.env.s.copy())
                action = self.agent.act(state)
                next_state, reward, done = self.env.step_action(action)
                episode_reward.append(reward)
                self.agent.memorize(state, action, reward, next_state, done)
                state = next_state
                actions.append(action)
                if done:
                    print('         - SOLVED! episode {}/{} steps taken: {} reward: {}'.format(episode, episodes,
                                                                                               len(actions), reward))
                    steps_taken_for_completion.append(len(actions))
                    solved_eps = solved_eps + 1
                    actions_enum = []
                    for i in actions:
                        actions_enum.append(Action(i))
                    self.agent.update_target_model()
                    break
                if len(self.agent.memory) > 10:
                    self.agent.replay(5)
            if not done:
                steps_taken_for_completion.append(steps)
            #self.predictActions()
            episode_rewards.append(episode_reward)
        return solved_eps, steps_taken_for_completion, episode_rewards

    def predictActions(self):
        map = np.zeros((5, 5))
        row = 0
        col = 0
        agent_pos = self.env.start_state
        goal_pos = self.env.terminal_state
        for i in range(self.env.dimension):
            for j in range(self.env.dimension):
                map[agent_pos[0]][agent_pos[1]] = 0
                agent_pos = (row, col)
                print(agent_pos)
                if agent_pos != goal_pos:
                    map[goal_pos[0]][goal_pos[1]] = 2

                map[agent_pos[0]][agent_pos[1]] = 1

                prediction = self.agent._predict(map)

                for i in range(len(map)):
                    print("             {}".format(map[i]))

                print("             {}".format(prediction))
                action_taken = Action(np.argmax(prediction[0]))
                print("             CHOSEN ACTION: {}".format(action_taken.name))
                print("\n")
                col = col + 1
            row = row + 1
            col = 0

    def run_experiment_eps(self, episodes, iterations):
        episode_results = []
        samples_results = {}
        steps_taken = []
        start_positions = []
        goal_positions = []
        for it in range(iterations):
            print("iteration", it)
            total_solved, steps_taken_for_completion, episode_reward = self.compute_iteration(episodes,
                                                                                            Constants.MAX_STEPS)
            steps_taken.append(steps_taken_for_completion)
            episode_results.append(EpisodeResult(episodes, steps_taken_for_completion, total_solved,
                                                 episode_reward, self.env.distance_from_start_to_goal))
            print("episodes: {} total_solved: {} total reward: {}".format(episodes, total_solved, episode_reward))
            print("\n\n")
            samples_results[it] = episode_results
            episode_results = []
            start_positions.append(self.env.start_state)
            goal_positions.append(self.env.terminal_state)
        return samples_results, start_positions, goal_positions

    def run_experiment_change_location(self, episodes, iterations, changeOrigin):
        changeLocation = False
        episode_results = []
        samples_results = {}
        steps_taken = []
        start_positions = []
        goal_positions = []
        for it in range(iterations):
            if np.random.rand() <= 0.5:
                changeLocation = True
            if changeLocation:
                if changeOrigin:
                    self.changeOrigin()
                else:
                    self.changeGoal()
            print("iteration", it)
            total_solved, steps_taken_for_completion, episode_reward = self.compute_iteration(episodes,
                                                                                            Constants.MAX_STEPS)
            steps_taken.append(steps_taken_for_completion)
            episode_results.append(EpisodeResult(episodes, steps_taken_for_completion, total_solved,
                                                 episode_reward, self.env.distance_from_start_to_goal))
            print("episodes: {} total_solved: {} total reward: {}".format(episodes, total_solved, episode_reward))
            print("\n\n")
            samples_results[it] = episode_results
            episode_results = []
            changeLocation = False
            self.agent.save_model("model_it_{}".format(it), False)
            start_positions.append(self.env.start_state)
            goal_positions.append(self.env.terminal_state)
        return samples_results, start_positions, goal_positions

    def changeOrigin(self):
        old_distance = self.env.distance_from_start_to_goal
        start_state = self.env.start_state
        new_start_state = self.getNewLocation()
        print("old origin state: {} new origin state: {}".format(start_state, new_start_state))
        self.env.changeStartState(new_start_state)
        print("Old distance: {} New distance: {}".format(old_distance, self.env.distance_from_start_to_goal))

    def changeGoal(self):
        old_distance = self.env.distance_from_start_to_goal
        terminal_state = self.env.terminal_state
        new_terminal_state = self.getNewLocation()
        print("old terminal state: {} new terminal state: {}".format(terminal_state, new_terminal_state))
        self.env.changeTerminalState(new_terminal_state)
        print("Old distance: {} New distance: {}".format(old_distance, self.env.distance_from_start_to_goal))

    def getNewLocation(self):
        terminal_state = self.env.terminal_state
        start_state = self.env.start_state
        col = randint(0, self.env.dimension - 1)
        row = randint(0, self.env.dimension - 1)
        while ((row, col) == terminal_state) or ((row, col) == start_state):
            col = randint(0, self.env.dimension - 1)
            row = randint(0, self.env.dimension - 1)
        return row, col

