from datetime import datetime
from os import listdir
from os.path import isfile, join

import gym
import os
import pandas as pd
import numpy as np

from DQNAgent import DQNAgent
from Domain.ExperimentType import ExperimentType
from Infrastructure.ExperimentService import ExperimentService

import Presentation.GraphService as graphService


def create_environment():
    print('Creating environment')


class Controller:

    def __init__(self, episodes, iterations, experiment_type):
        self.env = self.createEnvironment()
        self.agent, self.use_existing_model = self.createAgent(self.env)
        self.episodes = episodes
        self.iterations = iterations
        self.experiment_type = ExperimentType(experiment_type)
        self.experimentService = ExperimentService(self.env, self.agent)

    def createEnvironment(self):
        env = gym.make("env:MatrixEnv-v0")
        env.init_variables(5, (0, 0), (1, 4))
        return env

    def createAgent(self, env):
        use_existing_model = False
        state_size = env.observation_space.n
        action_size = env.action_space.n
        total_models = len(os.listdir(os.getcwd() + '/model_old'))
        if total_models != 0:
            use_saved_model = input('Another model already exists, use existing model? y/n: ')
            if (use_saved_model == 'y') or (use_saved_model == 'Y'):
                use_existing_model = True
                onlyfiles = [f for f in listdir(os.getcwd() + '/model_old') if
                             isfile(join(os.getcwd() + '/model_old', f))]
                print(onlyfiles)
                requested_model = input('Enter the model you want to use from the saved models: ')
                created_agent = False
                while not created_agent:
                    if requested_model in onlyfiles:
                        _agent = DQNAgent(state_size, action_size, True, requested_model)
                        created_agent = True
                    else:
                        print("Model does not exists. \n")
                        requested_model = input('Enter the model you want to use from the saved models: ')
            else:
                _agent = DQNAgent(state_size, action_size, False, '')
        else:
            _agent = DQNAgent(state_size, action_size, False, '')
        self.agent = _agent
        return _agent, use_existing_model

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
            self.save_model_results(samples_results)

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

    def run_experiment(self):
        samples_results = {}
        if self.experiment_type == ExperimentType.EPISODES:
            samples_results = self.experimentService.run_experiment(self.episodes, self.iterations)
        elif self.experiment_type == ExperimentType.CHANGE_DIMENSION:
            samples_results = self.experimentService.run_experiment(self.episodes, self.iterations)
        elif self.experiment_type == ExperimentType.CHANGE_GOAL:
            samples_results = self.experimentService.run_experiment(self.episodes, self.iterations)
        elif self.experiment_type == ExperimentType.CHANGE_ORIGIN:
            samples_results = self.experimentService.run_experiment(self.episodes, self.iterations)
        elif self.experiment_type == ExperimentType.DISABLE_TILE:
            samples_results = self.experimentService.run_experiment(self.episodes, self.iterations)
        self.save_experiment_result(samples_results, self.use_existing_model)

    def save_model_results(self, samples_results):
        now = datetime.now()
        file_name = "{}-{}-{}".format(self.agent.requested_model, self.experiment_type.name, now.strftime("%d_%m-%H_%M"))
        print(file_name)
        self.saveExperiment(self.agent.requested_model,
                            ExperimentType(0),
                            self.agent.requested_model,
                            self.episodes,
                            self.iterations,
                            samples_results,
                            file_name + ".pkl")
