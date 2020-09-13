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
        if dimension != 0 and dimension != -1:
            self.env = self.createEnvironment(dimension, (0, 0), (7, 7))
            self.agent, self.use_existing_model = self.createAgent(self.env)
            self.agent.save_model(self.new_model_name, False)
            self.episodes = episodes
            self.iterations = iterations
            self.experiment_type = ExperimentType(experiment_type)
            self.experimentService = ExperimentService(self.env, self.agent)
        elif dimension == -1:
            dimension, origin, goal, requested_model, requested_model_path, \
            experiment_type, episodes, iterations, df = self.createElementsFromExisting()
            self.experiment_type = experiment_type
            self.episodes = episodes
            self.iterations = iterations
            self.env = self.createEnvironment(dimension, origin, goal)
            self.agent = DQNAgent(self, self.env.observation_space.n, self.env.action_space.n, True, requested_model,
                                          requested_model_path)
            new_model = datetime.now().strftime("%d-%m_%H-%M-%S")
            self.new_model_name = new_model
            self.experimentService = ExperimentService(self.env, self.agent)
            self.requested_model_path = requested_model_path
            self.use_existing_model = True
            self.redoExperiment(df)

    def redoExperiment(self, df):
        iterations_list = []
        total_iterations = 0
        for index, row in df.iterrows():
            origin = row['start_pos']
            goal = row['goal_pos']
            iterations_list.append((origin, goal))
            total_iterations = total_iterations + 1
        idx = 0
        for iteration_positions in iterations_list:
            print("{} - Initial position: {} Goal position: {}".format(idx, iteration_positions[0], iteration_positions[1]))
            idx = idx + 1

        case_order = []
        for idx in range(total_iterations):
            case_correct = False
            case = -1
            while not case_correct:
                case = input("Enter idx of next case: ")

                if case.isdigit() and int(case) in range(0, total_iterations) and int(case) not in case_order:
                    case_correct = True
                else:
                    print("Invalid index. Might not exist or has already been used. \n\n")
            case_order.append(int(case))
        self.redoRunExperiment(case_order, iterations_list)

    def redoRunExperiment(self, case_order, case_positions):
        samples_results = {}
        start_positions = []
        goal_positions = []
        idx = 0
        for case in case_order:
            start_position = case_positions[case][0]
            goal_position = case_positions[case][1]
            self.env.changeStartState(start_position)
            self.env.changeTerminalState(goal_position)
            samples_results_aux, start_positions_aux, goal_positions_aux = self.experimentService.run_experiment_eps(
                self.episodes, 1)
            samples_results[idx] = samples_results_aux
            start_positions.append(start_position)
            goal_positions.append(goal_position)
            self.agent.save_model("model_it_{}".format(idx), False)
            idx = idx + 1
            order_string = str(case_order).strip('[]')
        df = self.save_experiment_result(samples_results, start_positions, goal_positions, False, self.requested_model_path,
                                         order_string.strip(','))
        if not df.empty:
            self.show_experiment_results(df, True)


    def createElementsFromExisting(self):
        onlyfiles = [f for f in listdir(os.getcwd() + '/model')]
        print(onlyfiles)
        requested_model = input('Enter the model you want to use from the saved models: ')
        onlyfiles = [f for f in listdir(os.getcwd() + "/model/{}".format(requested_model))
                     if os.path.isdir(os.path.join(os.getcwd() + "/model/{}".format(requested_model), f))]
        print(onlyfiles)
        experiment = input('Choose experiment you want to redo: ')
        df = pickle.load(
            open(os.getcwd() + "/model/{}/{}/{}.pkl".format(requested_model, experiment, experiment), 'rb'))
        print(df)
        for index, row in df.iterrows():
            requested_model = row['modelUsed']
            dimension = row['dimension']
            origin = row['start_pos']
            print('origin {}'.format(origin))
            goal = row['goal_pos']
            print('goal {}'.format(goal))
            experiment_type = row['experimentType']
            episodes = row['episodes']
            iterations = row['iterations']
            requested_model_path = "model/{}/{}/{}".format(requested_model, experiment, 'initial_model')
            break
        return dimension, origin, goal, requested_model, requested_model_path, experiment_type, episodes, iterations, df



    def createEnvironment(self, dimension, origin=(-1, -1), goal=(-1, -1)):
        env = gym.make("env:MatrixEnv-v0")
        if origin == (-1, -1) and goal == (-1, -1):
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
                        _agent = DQNAgent(self, state_size, action_size, True, requested_model,
                                          "model/{}/{}".format(requested_model, requested_model))
                        created_agent = True
                    else:
                        print("Model does not exists. \n")
                        requested_model = input('Enter the model you want to use from the saved models: ')
            else:
                _agent = DQNAgent(self, state_size, action_size, False)
        else:
            _agent = DQNAgent(self, state_size, action_size, False)
        new_model = datetime.now().strftime("%d-%m_%H-%M-%S")
        self.new_model_name = new_model
        return _agent, use_existing_model

    def listExistingModels(self):
        total_models = len(os.listdir(os.getcwd() + '/model'))
        onlyfiles = [f for f in listdir(os.getcwd() + '/model')]

    def saveExperiment(self, parentDirectory, experimentType, modelUsed, episodes, iterations, dimension, start_pos,
                       goal_pos, episode_results, file_name, requested_model_path=''):
        dictlist = []
        for key, value in episode_results.items():
            temp = [key, value]
            dictlist.append(temp)

        d = {
            'experimentType': experimentType,
            'modelUsed': modelUsed,
            'episodes': episodes,
            'iterations': iterations,
            'episode_results': dictlist,
            'dimension': dimension,
            'start_pos': start_pos,
            'goal_pos': goal_pos
        }
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
        if self.experiment_type == ExperimentType.EPISODES and requested_model_path == '':
            path = "./model/{}".format(modelUsed[0])
        elif requested_model_path != '':
            path = requested_model_path
        else:
            path = "./model/{}/{}".format(modelUsed[0], parentDirectory)
        df.to_pickle("{}/{}".format(path, file_name))
        return df, path

    def group_models(self, path, requested_model_path=''):
        if requested_model_path == '':
            initial_model_name = "model/{}".format(self.new_model_name)
            os.rename(initial_model_name, "{}/{}".format(path, "initial_model"))
        if self.experiment_type != ExperimentType.EPISODES:
            if requested_model_path != '':
                path = requested_model_path
            for it in range(self.iterations):
                os.rename("model/model_it_{}".format(it),
                          "{}/{}".format(path, "{}_iter_{}".format(self.new_model_name, it)))

    def save_experiment_result(self, samples_results, start_positions, goal_positions, use_existing_model=False, requested_model_path='',
                               case_order=''):
        if requested_model_path == '':
            if use_existing_model:
                overwrite_model = input("Overwrite model? y/n : ")
                if (overwrite_model == 'y') or (overwrite_model == 'Y'):
                    self.agent.save_model(self.agent.requested_model, True)
            else:
                save_model = input("Save user model? y/n : ")
                if (save_model == 'y') or (save_model == 'Y'):
                    model_file_name = input('Enter file name: ')
                    self.agent.save_model(model_file_name, True)

        save_experiment = input("Save experiment? y/n : ")

        if (save_experiment == 'y') or (save_experiment == 'Y'):
            df, path = self.save_model_results(samples_results, start_positions, goal_positions,
                                               requested_model_path.replace('/initial_model', ''), case_order)
            self.group_models(path, requested_model_path.replace('/initial_model', ''))
            return df


    def show_experiment_results(self, df, redoExperiment):
        show_results = input("Show experiment results? y/n: ")
        if (show_results == 'y') or (show_results == 'Y'):
            graphService.plot_model_results(df, '', redoExperiment)

    def run_experiment(self):
        samples_results = {}
        start_positions = []
        goal_positions = []
        if self.experiment_type == ExperimentType.EPISODES:
            samples_results, start_positions, goal_positions = self.experimentService.run_experiment_eps(self.episodes,
                                                                                                         self.iterations)
        elif self.experiment_type == ExperimentType.CHANGE_GOAL:
            samples_results, start_positions, goal_positions = self.experimentService.run_experiment_change_location(
                self.episodes, self.iterations, False)
        elif self.experiment_type == ExperimentType.CHANGE_ORIGIN:
            samples_results, start_positions, goal_positions = self.experimentService.run_experiment_change_location(
                self.episodes, self.iterations, True)
        df = self.save_experiment_result(samples_results, start_positions, goal_positions, self.use_existing_model)
        if not df.empty:
            self.show_experiment_results(df, False)

    def save_model_results(self, samples_results, start_positions, goal_positions, requested_model_path= '', case_order=''):
        now = datetime.now()
        if case_order:
            file_name = "{}-{}-{}".format(self.experiment_type.name, now.strftime("%d_%m-%H_%M"), case_order)
        else:
            file_name = "{}-{}".format(self.experiment_type.name, self.new_model_name)
        if self.experiment_type != ExperimentType.EPISODES and requested_model_path == '':
            parent_directory = os.getcwd()
            model_directory = "model/{}".format(self.agent.requested_model)
            path = os.path.join(parent_directory, model_directory)
            if not os.path.isdir(os.path.join(path, file_name)):
                os.mkdir(os.path.join(path, file_name))
            parent_path = file_name
        elif requested_model_path != '':
            parent_path = requested_model_path
        else:
            parent_path = self.agent.requested_model

        return self.saveExperiment(parent_path,
                                   [ExperimentType(self.experiment_type)] * self.iterations,
                                   [self.agent.requested_model] * self.iterations,
                                   [self.episodes] * self.iterations,
                                   [self.iterations] * self.iterations,
                                   [self.env.dimension] * self.iterations,
                                   start_positions,
                                   goal_positions,
                                   samples_results,
                                   file_name + ".pkl",
                                   requested_model_path)

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
        experimentName = input("Enter experiment: ")
        if ".pkl" in experimentName:
            experimentName = experimentName.replace(".pkl", "")

        fileName = experimentName + ".pkl"

        experimentFiles = []

        if os.path.isdir(os.getcwd() + "/model/{}/{}".format(modelName, experimentName)):
            experimentFiles = os.listdir(os.getcwd() + "/model/{}/{}".format(modelName, experimentName))

        showExperiment = True

        while showExperiment:
            if len(experimentFiles) > 1:
                experimentFiles.sort()
                for experiment in experimentFiles:
                    if '.pkl' in experiment:
                        print(experiment)
                fileName = input("Enter experiment you want to see: ")
                f = open(os.getcwd() + "/model/{}/{}/{}".format(modelName, experimentName, fileName), 'rb')
            else:
                f = open(os.getcwd() + "/model/{}/{}".format(modelName, fileName), 'rb')

            graphService.plot_model_results(pickle.load(f), fileName, experimentName + ".pkl" != fileName)
            if len(experimentFiles) > 1:
                showExperiment = input("Show another experiment?: ") == 'y'
            else:
                showExperiment = False


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
