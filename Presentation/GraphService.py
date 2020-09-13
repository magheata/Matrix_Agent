import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def plot_boxplot(data, title, xLabel, yLabel, colors):
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    bplot1 = ax1.boxplot(data, flierprops=green_diamond, patch_artist=True)
    bplot = bplot1['boxes']
    for patch, color in zip(bplot, colors):
        patch.set_facecolor(color)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()


def plot_graph(total_episodes, data_list, colors, labels, title, xLabel, yLabel):
    fontP = FontProperties()
    fontP.set_size('small')
    idx = 0
    for data in data_list:
        plt.plot(total_episodes, data, color=colors[idx], label=labels[idx])
        idx = idx + 1
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.title(title)
    plt.show()


def plot_model_results(df, fileName, redoExperiment):
    reward_steps = []
    reward_steps_list = []
    steps_array_list = []
    colors = []
    labels = []
    labels_idx = 1
    total_episodes = 0
    for index, row in df.iterrows():
        if redoExperiment:
            episode_result = row['episode_results'][1][0][0]
        else:
            episode_result = row['episode_results'][1][0]
        for reward in episode_result.episode_reward:
            if isinstance(reward, int):
                reward_steps.append(reward)
            else:
                reward_steps.append(reward[-1])
        reward_steps_list.append(reward_steps)
        steps_array_list.append(np.array(episode_result.steps_to_completion))
        total_episodes = episode_result.episodes
        colors.append(np.random.rand(3, ))
        labels.append("Caso {}".format(labels_idx))
        labels_idx = labels_idx + 1
        reward_steps = []

    iterations_array = np.arange(start=1, stop=total_episodes + 1)

    plot_graph(iterations_array, steps_array_list, colors, labels, 'Acciones realizadas en cada episodio \n{}'.format(fileName),
               'Episodio', 'Media de acciones realizadas por episodio')
    plot_graph(iterations_array, reward_steps_list, colors, labels, 'Recompensa final de cada episodio \n{}'.format(fileName),
               'Episodio', 'Recompensa')
    plot_boxplot(steps_array_list, "Acciones realizadas \n{}".format(fileName), "Caso", "Acciones realizadas en cada episodio",
                 colors)
