import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import statistics

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

def plot_piechart(data, title, labels):
    fontP = FontProperties()
    fontP.set_size('small')
    fig1, ax1 = plt.subplots()
    slices, _, _ = ax1.pie(data, colors=('#db504a', '#55A630'), autopct=lambda p : '{:.2f}%\n({:,.0f} episodios)'.format(p,p * sum(data)/100), startangle=90)
    # draw circle
    plt.legend(slices, labels, loc="best")
    centre_circle = plt.Circle((0, 0), 0.80, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
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
    labels_piechart = 'Objetivo no alcanzado', 'Objetivo alcanzado'
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
        action_combinations = {k: v for k, v in episode_result.action_combinations.items() if v >= 5}

        sorted_actions_combinations = {k: v for k, v in sorted(action_combinations.items(), key=lambda item: item[1], reverse=True)}
        for key in sorted_actions_combinations.keys():
            print("agent_pos: {} goal_pos: {}  {}: {} \n".format(row['start_pos'], row['goal_pos'], key,
                                                                 sorted_actions_combinations[key]))
        steps_array_list.append(np.array(episode_result.steps_to_completion))
        total_episodes = episode_result.episodes
        colors.append(np.random.rand(3, ))
        labels.append("Caso {}".format(labels_idx))
        labels_idx = labels_idx + 1
        print("Media recompensas: {}".format(statistics.mean(reward_steps)))
        print("Std dev: {}".format(statistics.stdev(reward_steps)))

        reward_steps = []
        print(episode_result.action_combinations)
        print("Distance to goal: {}".format(episode_result.distance_to_goal))
        print('Goal pos: {} Start pos: {}'.format(row['goal_pos'], row['start_pos']))
        print("Total resueltos: {}".format(episode_result.total_solved))

        plot_piechart([total_episodes - episode_result.total_solved, episode_result.total_solved],
                      'Porcentage de episodios en los que el agente ha ' +
                      'alcanzado el objetivo', labels_piechart)

    iterations_array = np.arange(start=1, stop=total_episodes + 1)

    plot_graph(iterations_array, steps_array_list, colors, labels, 'Acciones realizadas en cada episodio',
               'Episodio', 'Media de acciones realizadas por episodio')
    plot_graph(iterations_array, reward_steps_list, colors, labels, 'Recompensa final de cada episodio',
               'Episodio', 'Recompensa')
    plot_boxplot(steps_array_list, "Acciones realizadas", "Caso", "Acciones realizadas en cada episodio",
                 colors)

