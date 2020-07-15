from Action import Action
from DQNAgent import DQNAgent

import tkinter as tk

import os
import matplotlib.pyplot as plt
import gym
import seaborn as sns

import GraphService as graphService

import tensorflow as tf
from tensorflow import keras
from os import listdir
from os.path import isfile, join
from Domain.EpisodeResult import EpisodeResult

EPISODES = 50

TOTAL_SAMPLES = 10

def plot(solved_episodes):
    plt(EPISODES * TOTAL_SAMPLES, solved_episodes)
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


def compute_episodes(total_episodes):
    solved_eps = 0
    steps_taken_for_completion = []
    for episode in range(total_episodes):
        actions = []
        env_states = []
        state = env.reset_action()
        print('episode {}/{}'.format(episode, total_episodes))
        for time in range(20):
            env_states.append(env.s.copy())
            action = agent.act(state)
            next_state, reward, done = env.step_action(action, len(actions))
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            actions.append(action)
            if done:
                print('         - SOLVED! episode {}/{} steps taken: {}'.format(episode, total_episodes, len(actions)))
                steps_taken_for_completion.append(len(actions))
                solved_eps = solved_eps + 1
                actions_enum = []
                for i in actions:
                    actions_enum.append(Action(i))
                agent.update_target_model()
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    return solved_eps, steps_taken_for_completion


if __name__ == "__main__":

    window = tk.Tk()
    label = tk.Label(text="Python rocks!")
    label.pack()
    window.mainloop()
    env = gym.make("env:MatrixEnv-v0")
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
                    agent = DQNAgent(state_size, action_size, True, 'model/' + requested_model)
                    created_agent = True
                else:
                    print("Model does not exists. \n")
                    requested_model = input('Enter the model you want to use from the saved models: ')
        else:
            agent = DQNAgent(state_size, action_size, False, '')
    else:
        agent = DQNAgent(state_size, action_size, False, '')

    #print(agent.model.get_weights())
    batch_size = 32
    episode_results = []
    samples_results = {}

    steps_taken = []
    for sample in range(TOTAL_SAMPLES):
        SOLVED_TOTAL = []
        print("sample", sample)
        total_solved, steps_taken_for_completion = compute_episodes(EPISODES)

        steps_taken.append(steps_taken_for_completion)

        episode_results.append(EpisodeResult(EPISODES, steps_taken_for_completion, total_solved))

        print("episodes: {} total_solved: {}".format(EPISODES, total_solved))
        print("\n\n")
        samples_results[sample] = episode_results
        episode_results = []

    graphService.create_boxplot_actions(steps_taken)

    save_model = input("Save user model? y/n : ")

    if (save_model == 'y') or (save_model == 'Y'):
        model_file_name = input('Enter file name: ')
        agent.save_model('model/' + model_file_name)


