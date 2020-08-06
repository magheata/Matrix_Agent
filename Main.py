import matplotlib.pyplot as plt
import numpy as np

from Domain.Action import Action
from Application.Controller import Controller
from Domain.ExperimentType import ExperimentType

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


if __name__ == "__main__":

    episodes_correct = False
    episodes = -1
    while not episodes_correct:
        episodes = input("Total episodes for this experiment: ")
        if episodes.isdigit():
            episodes_correct = True
        else:
            print("Positive number of episodes required\n\n")

    iterations_correct = False
    iterations = -1
    while not iterations_correct:
        iterations = input("Total iterations/episode for this experiment: ")
        if iterations.isdigit():
            iterations_correct = True
        else:
            print("Positive number of iterations required\n\n")

    dimension_correct = False
    dimension = -1
    while not dimension_correct:
        dimension = input("Dimension of the matrix: ")
        if dimension.isdigit():
            dimension_correct = True
        else:
            print("Positive dimension required\n\n")

    for experiment in (ExperimentType):
        print("{} - {}".format(experiment.value, experiment.name))

    experiment_type = input("Choose the type of experiment (enter index): ")

    controller = Controller(int(dimension), int(episodes), int(iterations), int(experiment_type))
    controller.run_experiment()
    #predictActions(env, agent)
    #
