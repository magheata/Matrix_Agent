import matplotlib.pyplot as plt
import numpy as np

import Constants
from Domain.HistogramEpisode import HistogramEpisode


def create_boxplot_actions(steps_taken):
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig3, ax3 = plt.subplots()
    ax3.set_title('Acciones realizadas para llegar a la solución')
    ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Number of actions made')

    ax3.boxplot(steps_taken, flierprops=green_diamond)
    plt.show()

def plot_mean_steps(total_iterations, mean_steps, distance_to_goal):
    plt.plot(total_iterations, mean_steps, color='g')
    #if len(distance_to_goal) > 0:
    #    plt.plot(total_iterations, ones(size(distance_to_goal, color='r')
    plt.xlabel('Iteration')
    plt.ylabel('Mean steps taken')
    plt.title('Steps taken in each iteration to reach goal')
    plt.show()


def plot_variance_steps(total_iterations, variance_steps):
    plt.plot(total_iterations, variance_steps, color='g')
    plt.xlabel('Iteration')
    plt.ylabel('Variance of the steps taken')
    plt.title('Variance of steps taken to reach goal')
    plt.show()


def plot_std_dev_steps(total_iterations, std_dev_steps):
    plt.plot(total_iterations, std_dev_steps, color='g')
    plt.xlabel('Episode')
    plt.ylabel('Standard deviation of the steps taken')
    plt.title('Std deviation of the steps taken to reach goal')
    plt.show()


def plot_reward(total_iterations, reward_results):
    plt.plot(total_iterations, reward_results, color='g')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Final reward in each iteration')
    plt.show()


def plot_error_steps(total_iterations, error_steps_results):
    plt.plot(total_iterations, error_steps_results, color='g')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.title('Steps taken to reach goal in each episode')
    plt.show()


def plot_model_results(df):
    mean_steps = []
    variance_steps = []
    std_var_steps = []
    reward_steps = []
    error_steps = []
    distance_to_goal = []
    total_episodes = 0
    for index, row in df.iterrows():
        episode_result = row['episode_results'][1][0]
        for reward in episode_result.episode_reward:
            if isinstance(reward, int):
                reward_steps.append(reward)
            else:
                reward_steps.append(reward[-1])

        print(reward)
        steps_array = np.array(episode_result.steps_to_completion)
        total_episodes = episode_result.episodes
        _error_steps = []
        mean_steps.append(steps_array.mean())
        variance_steps.append(steps_array.var())
        std_var_steps.append(steps_array.std())
        if hasattr(episode_result, 'distance_to_goal'):
            for step in steps_array:
                error = episode_result.distance_to_goal - step
                _error_steps.append(step)
            distance_to_goal.append(episode_result.distance_to_goal)

    iterations_array = np.arange(start=1, stop=total_episodes + 1)

    plot_mean_steps(iterations_array, steps_array, distance_to_goal)
    # plot_reward(len(df), reward_results)
    #plot_std_dev_steps(iterations_array, std_var_steps)
    #plot_variance_steps(iterations_array, variance_steps)
    plot_reward(iterations_array, reward_steps)
    #if hasattr(episode_result, 'distance_to_goal'):
    #    plot_error_steps(iterations_array, error_steps)

def plot_model_results_old(df):
    mean_steps = []
    variance_steps = []
    std_var_steps = []
    reward_steps = []
    error_steps = []
    distance_to_goal = []
    total_episodes = 0
    for index, row in df.iterrows():
        episode_result = row['episode_results'][1][0]
        reward_steps.append(episode_result.episode_reward[-1])
        steps_array = np.array(episode_result.steps_to_completion)
        total_episodes = episode_result.iterations
        if len(steps_array) is not 0:
            _error_steps = []
            mean_steps.append(steps_array.mean())
            variance_steps.append(steps_array.var())
            std_var_steps.append(steps_array.std())
            if hasattr(episode_result, 'distance_to_goal'):
                for step in steps_array:
                    error = episode_result.distance_to_goal - step
                    _error_steps.append(step)
                error_steps.append(np.array(_error_steps).mean())
        else:
            mean_steps.append(0)
            variance_steps.append(0)
            std_var_steps.append(0)
            if hasattr(episode_result, 'distance_to_goal'):
                error_steps.append(episode_result.distance_to_goal)
        if hasattr(episode_result, 'distance_to_goal'):
            distance_to_goal.append(episode_result.distance_to_goal)
    iterations_array = np.arange(start=1, stop=200 + 1)

    plot_mean_steps(iterations_array, mean_steps, distance_to_goal)
    #plot_reward(len(df), reward_results)
    plot_std_dev_steps(iterations_array, std_var_steps)
    plot_variance_steps(iterations_array, variance_steps)
    plot_reward(iterations_array, reward_steps)
    if hasattr(episode_result, 'distance_to_goal'):
        plot_error_steps(iterations_array, error_steps)