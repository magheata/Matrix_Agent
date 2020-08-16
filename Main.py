import matplotlib.pyplot as plt
import numpy as np

from Domain.Action import Action
from Application.Controller import Controller
from Domain.ExperimentType import ExperimentType

ITERATIONS = 1

EPISODES = 1


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


def runNewExperiment():
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

    for experiment in ExperimentType:
        print("{} - {}".format(experiment.value, experiment.name))

    experiment_type = input("Choose the type of experiment (enter index): ")

    controller = Controller(int(dimension), int(episodes), int(iterations), int(experiment_type))
    controller.run_experiment()


if __name__ == "__main__":

    execType_correct = False
    execType = -1
    while not execType_correct:
        execType = input("Choose execution type: \n 0 - New experiment \n 1 - Show experiment results \n")
        if execType.isdigit() or execType != 0 or execType != 1:
            execType_correct = True
        else:
            print("Invalid type, choose again. \n\n")

    if int(execType) == 0:
        runNewExperiment()
    else:
        controller = Controller()
        controller.readExperiment()
    #predictActions(env, agent)
    #
