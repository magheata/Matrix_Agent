import matplotlib.pyplot as plt
import numpy as np

def plot_mean_steps(total_iterations, mean_steps):
    plt.plot(total_iterations, mean_steps, color='g')
    plt.xlabel('Episode')
    plt.ylabel('Mean steps taken')
    plt.title('Steps taken in each episode')
    plt.show()

def plot_reward(total_iterations, reward_results):
    plt.plot(total_iterations, reward_results, color='g')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Final reward in each episode')
    plt.show()

def plot_model_results(df):
    mean_steps = []
    variance_steps = []
    std_var_steps = []
    reward_steps = []
    total_episodes = 0
    for index, row in df.iterrows():
        episode_result = row['episode_results'][1][0]
        for reward in episode_result.episode_reward:
            if isinstance(reward, int):
                reward_steps.append(reward)
            else:
                reward_steps.append(reward[-1])

        print(reward_steps)
        steps_array = np.array(episode_result.steps_to_completion)
        total_episodes = episode_result.episodes
        _error_steps = []
        mean_steps.append(steps_array.mean())
        variance_steps.append(steps_array.var())
        std_var_steps.append(steps_array.std())

    iterations_array = np.arange(start=1, stop=total_episodes + 1)

    plot_mean_steps(iterations_array, steps_array)
    plot_reward(iterations_array, reward_steps)
