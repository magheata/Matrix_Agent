from os import listdir
from os.path import isfile, join

import gym
import os

from DQNAgent import DQNAgent
from Presentation.Window import Window


def create_environment():
    print('Creating environment')


class Controller:

    def setSamples(self, samples):
        self.samples = samples

    def setIterations(self, iterations):
        self.iterations = iterations

    def startComputing(self):
        if (self.iterations is not None) and (self.samples is not None):
            print('Okay')

    def initEnvironment(self):
        dimension = self.window.getDimension()
        self.env = gym.make("env:MatrixEnv-v0")
        self.env.init_variables(self.window.getDimension(),
                                self.window.getStartPosition(),
                                self.window.getGoalPosition())

    def createEnvironment(self):
        env = gym.make("env:MatrixEnv-v0")
        env.init_variables(5, (0, 0), (1, 4))
        return env

    def createAgent(self, env):
        state_size = env.observation_space.n
        action_size = env.action_space.n
        total_models = len(os.listdir(os.getcwd() + '/model'))
        if total_models != 0:
            use_saved_model = input('Another model already exists, use existing model? y/n: ')
            if (use_saved_model == 'y') or (use_saved_model == 'Y'):
                onlyfiles = [f for f in listdir(os.getcwd() + '/model') if isfile(join(os.getcwd() + '/model', f))]
                print(onlyfiles)
                requested_model = input('Enter the model you want to use from the saved models: ')
                created_agent = False
                while not created_agent:
                    if requested_model in onlyfiles:
                        _agent = DQNAgent(state_size, action_size, True, 'model/' + requested_model)
                        created_agent = True
                    else:
                        print("Model does not exists. \n")
                        requested_model = input('Enter the model you want to use from the saved models: ')
            else:
                _agent = DQNAgent(state_size, action_size, False, '')
        else:
            _agent = DQNAgent(state_size, action_size, False, '')
        return _agent