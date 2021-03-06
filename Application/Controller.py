from datetime import datetime
from os import listdir

import gym
import os
import pandas as pd
import numpy as np
import pickle

from random import randint

from DQNAgent import DQNAgent
from Domain.Action import Action
from Domain.ExperimentType import ExperimentType
from Infrastructure.ExperimentService import ExperimentService

import Presentation.GraphService as graphService


class Controller:

    def __init__(self, dimension=0, episodes=0, iterations=0, experiment_type=ExperimentType.EPISODES):
        if dimension != 0:
            self.env = self.createEnvironment(dimension)
            self.agent, self.use_existing_model = self.createAgent(self.env)
            self.episodes = episodes
            self.iterations = iterations
            self.experiment_type = ExperimentType(experiment_type)
            self.experimentService = ExperimentService(self.env, self.agent)

    def createEnvironment(self, dimension):
        env = gym.make("env:MatrixEnv-v0")
        origin = (randint(0, dimension - 1), randint(0, dimension - 1))
        goal = (randint(0, dimension - 1), randint(0, dimension - 1))
        env.init_variables(dimension, origin, goal)
        print(env.s)
        print("Distance from start to goal is: {}".format(env.distance_from_start_to_goal))
        return env

    def createAgent(self, env):
        use_existing_model = False
        state_size = env.observation_space.n
        action_size = env.action_space.n
        total_models = len(os.listdir(os.getcwd() + '/model'))
        if total_models != 0:
            use_saved_model = input('Another model already exists, use existing model? y/n: ')
            if (use_saved_model == 'y') or (use_saved_model == 'Y'):
                use_existing_model = True
                onlyfiles = [f for f in listdir(os.getcwd() + '/model')]
                print(onlyfiles)
                requested_model = input('Enter the model you want to use from the saved models: ')
                created_agent = False
                while not created_agent:
                    if requested_model in onlyfiles:
                        _agent = DQNAgent(self, state_size, action_size, True, requested_model)
                        created_agent = True
                    else:
                        print("Model does not exists. \n")
                        requested_model = input('Enter the model you want to use from the saved models: ')
            else:
                _agent = DQNAgent(self, state_size, action_size, False, '')
        else:
            _agent = DQNAgent(self, state_size, action_size, False, '')
        self.agent = _agent
        return _agent, use_existing_model

    def listExistingModels(self):
        total_models = len(os.listdir(os.getcwd() + '/model'))
        onlyfiles = [f for f in listdir(os.getcwd() + '/model')]

    def saveExperiment(self, parentDirectory, experimentType, modelUsed, episodes, iterations, episode_results,
                       file_name):
        dictlist = []
        for key, value in episode_results.items():
            temp = [key, value]
            dictlist.append(temp)

        d = {
            'experimentType': experimentType,
            'modelUsed': modelUsed,
            'episodes': episodes,
            'iterations': iterations,
            'episode_results': dictlist
        }
        df = pd.DataFrame(data=d)
        df.to_pickle("./model/{}/{}".format(parentDirectory, file_name))
        return df

    def save_experiment_result(self, samples_results, use_existing_model):
        if use_existing_model:
            overwrite_model = input("Overwrite model? y/n : ")
            if (overwrite_model == 'y') or (overwrite_model == 'Y'):
                self.agent.save_model(self.agent.requested_model)
        else:
            save_model = input("Save user model? y/n : ")
            if (save_model == 'y') or (save_model == 'Y'):
                model_file_name = input('Enter file name: ')
                self.agent.save_model(model_file_name)

        save_experiment = input("Save experiment? y/n : ")

        if (save_experiment == 'y') or (save_experiment == 'Y'):
            return self.save_model_results(samples_results)

    def plot_experiment_results(self, steps_taken):
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
        graphService.create_boxplot_actions(steps_taken)

    def show_experiment_results(self, df):
        show_results = input("Show experiment results? y/n: ")
        if (show_results == 'y') or (show_results == 'Y'):
            graphService.plot_model_results(df)

    def run_experiment(self):
        samples_results = {}
        if self.experiment_type == ExperimentType.EPISODES:
            samples_results = self.experimentService.run_experiment_eps(self.episodes, self.iterations)
        elif self.experiment_type == ExperimentType.CHANGE_GOAL:
            samples_results = self.experimentService.run_experiment_change_goal(self.episodes, self.iterations)
        elif self.experiment_type == ExperimentType.CHANGE_ORIGIN:
            samples_results = self.experimentService.run_experiment_change_origin(self.episodes, self.iterations)
        df = self.save_experiment_result(samples_results, self.use_existing_model)

        if not df.empty:
            self.show_experiment_results(df)

    def save_model_results(self, samples_results):
        now = datetime.now()
        file_name = "{}-{}-{}".format(self.agent.requested_model, self.experiment_type.name,
                                      now.strftime("%d_%m-%H_%M"))
        print(file_name)
        return self.saveExperiment(self.agent.requested_model,
                            ExperimentType(0),
                            self.agent.requested_model,
                            self.episodes,
                            self.iterations,
                            samples_results,
                            file_name + ".pkl")

    def readExperiment(self):
        onlyfiles = [f for f in listdir(os.getcwd() + '/model')]
        onlyfiles.sort()
        onlyfiles.remove('.DS_Store')
        for file in onlyfiles:
            print(file)
        modelName = input("Enter model: ")
        modelExperiments = [f for f in listdir(os.getcwd() + '/model' + '/' + modelName)]
        modelExperiments.sort()
        modelExperiments.remove(modelName)
        for experiment in modelExperiments:
            print(experiment)
        fileName = input("Enter experiment: ")
        with open(os.getcwd() + "/model/{}/{}".format(modelName, fileName), 'rb') as f:
            graphService.plot_model_results(pickle.load(f))

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