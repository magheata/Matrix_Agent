import json

import matplotlib.pyplot as plt
import GraphService as graphService
import numpy as np
import os

from Domain.Action import Action
from Application.Controller import Controller
from Domain.EpisodeResult import EpisodeResult
from Domain.ExperimentType import ExperimentType

from datetime import datetime

ITERATIONS = 1

EPISODES = 1


def plot(solved_episodes):
    plt(ITERATIONS * EPISODES, solved_episodes)
    plt.title('Total solved episodes', fontsize=14)
    plt.ylabel('Solved episodes', fontsize=14)
    plt.xlabel('Total episodes', fontsize=14)
    plt.grid(True)
    plt.show()


def print_samples_results(sample_results):
    for sample in sample_results.keys():
        episode_results = sample_results.get(sample)
        for episode_result in episode_results:
            print("sample {}, episode_result {}".format(sample, episode_result))
        print("\n\n")


def compute_episodes(total_episodes):
    solved_eps = 0
    total_reward = 0
    steps_taken_for_completion = []
    for episode in range(total_episodes):
        actions = []
        env_states = []
        state = env.reset_action()
        print('episode {}/{}'.format(episode, total_episodes))
        for time in range(20):
            env_states.append(env.s.copy())
            action = agent.act(state)
            next_state, reward, done = env.step_action(action, len(actions))
            total_reward = total_reward + reward
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            actions.append(action)
            if done:
                print('         - SOLVED! episode {}/{} steps taken: {}'.format(episode, total_episodes, len(actions)))
                steps_taken_for_completion.append(len(actions))
                solved_eps = solved_eps + 1
                actions_enum = []
                for i in actions:
                    actions_enum.append(Action(i))
                agent.update_target_model()
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    return solved_eps, steps_taken_for_completion, total_reward


def predictActions(env, agent):
    map = env.s
    agent_pos = env.start_state
    goal_pos = env.terminal_state
    for i in range(env.dimension):
        for j in range(env.dimension):
            map[agent_pos[0]][agent_pos[1]] = 0
            agent_pos = (i, j)

            if agent_pos != goal_pos:
                map[goal_pos[0]][goal_pos[1]] = 2

            map[agent_pos[0]][agent_pos[1]] = 1
            print(map)
            prediction = agent._predict(map)
            print(prediction)
            print(Action(np.argmax(prediction[0])))
            print("\n\n")


def save_model_results(model_name, experiment_type):
    now = datetime.now()
    file_name = "{}-{}-{}".format(model_name, experiment_type.name, now.strftime("%d_%m-%H_%M"))
    print(file_name)
    controller.saveExperiment(model_name,
                              ExperimentType(0),
                              model_name,
                              EPISODES,
                              ITERATIONS,
                              samples_results,
                              file_name + ".pkl")


if __name__ == "__main__":

    controller = Controller()
    # controller.createDatabaseConnection()
    env = controller.createEnvironment()
    agent = controller.createAgent(env)
    # predictActions(env, agent)

    batch_size = 32
    episode_results = []
    samples_results = {}
    reward_results = []
    steps_taken = []
    for sample in range(EPISODES):
        SOLVED_TOTAL = []
        print("sample", sample)
        total_solved, steps_taken_for_completion, total_reward = compute_episodes(ITERATIONS)

        steps_taken.append(steps_taken_for_completion)

        episode_results.append(EpisodeResult(ITERATIONS, steps_taken_for_completion, total_solved, total_reward))

        reward_results.append(total_reward)

        print("episodes: {} total_solved: {}".format(ITERATIONS, total_solved))
        print("\n\n")
        samples_results[sample] = episode_results
        episode_results = []

    mean_steps = []
    variance_steps = []
    std_var_steps = []
    for steps in steps_taken:
        steps_array = np.array(steps)
        if len(steps_array) is not 0:
            print('Steps taken: {}'.format(steps))
            mean_steps.append(steps_array.mean())
            variance_steps.append(steps_array.var())
            std_var_steps.append(steps_array.std())
        else:
            mean_steps.append(0)
            variance_steps.append(0)
            std_var_steps.append(0)

    iterations_array = np.arange(start=1, stop=len(steps_taken) + 1)
    graphService.plot_mean_steps(iterations_array, mean_steps)
    graphService.plot_variance_steps(iterations_array, variance_steps)
    graphService.plot_std_dev_steps(iterations_array, std_var_steps)
    graphService.plot_reward(iterations_array, reward_results)

    graphService.create_boxplot_actions(steps_taken)

    save_model = input("Save user model? y/n : ")

    if (save_model == 'y') or (save_model == 'Y'):
        model_file_name = input('Enter file name: ')
        agent.save_model(model_file_name)

    save_xperiment = input("Save experiment? y/n : ")

    if (save_xperiment == 'y') or (save_xperiment == 'Y'):
        save_model_results(agent.requested_model, ExperimentType.EPISODES)
