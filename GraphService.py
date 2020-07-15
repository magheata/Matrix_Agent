import matplotlib.pyplot as plt
import numpy as np


from Domain.HistogramEpisode import HistogramEpisode


def create_boxplot_actions(steps_taken):
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig3, ax3 = plt.subplots()
    ax3.set_title('Acciones realizadas para llegar a la solución')
    ax3.boxplot(steps_taken, flierprops=green_diamond)
    plt.show()


def plot_steps_taken(total_samples, total_episodes, sample_results):
    width = 0.35
    fig, ax = plt.subplots()

    results_for_iteration = {}

    for sample in sample_results.keys():
        episode_results = sample_results.get(sample)
        for episode_result in episode_results:
            ide = np.arange(episode_result.iterations)

            actions_list = []
            for i in range(episode_result.iterations):
                if i in episode_result.steps_to_completion:
                    actions_list.append(episode_result.steps_to_completion[i])
                else:
                    actions_list.append(0)
            results_for_iteration[sample, episode_result.iterations] = HistogramEpisode(ide, sample, actions_list)

        print("\n\n")

    offset = 0
    ax.set_title('Total actions taken for completion [{} iterations]'.format(total_episodes))
    for j in range(total_samples):
        result_for_iteration = results_for_iteration[j, total_episodes]
        offset = offset + 1
        ax.bar(result_for_iteration.ide + (width * offset), result_for_iteration.list_steps_to_completion,
               label='sample #{}'.format(j))
    ax.legend()
    ax.autoscale_view()
    plt.figure()
    plt.pause(0.01)
    plt.show()
