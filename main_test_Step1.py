from datetime import datetime
from os import listdir

import gym
import os

import numpy as np
import pickle

from random import randint

from DQNAgent import DQNAgent
from Domain.Action import Action
from Domain.ExperimentType import ExperimentType
from Infrastructure.ExperimentService import ExperimentService

from numpy.random import seed
import tensorflow
seed(10)
import collections



if __name__ == "__main__":
    env = gym.make("env:MatrixEnv-v0")
    origin = (0, 0)
    goal = (0, 3)
    # origin = (randint(0, dimension - 1), randint(0, dimension - 1))
    # goal = (randint(0, dimension - 1), randint(0, dimension - 1))
    env.init_variables(4, origin, goal)
    print(env.s)
    print("Distance from start to goal is: {}".format(env.distance_from_start_to_goal))

    state_size = env.observation_space.n
    print("State_size: ",state_size)
    action_size = env.action_space.n
    print("Action size: ",action_size)

    agent = DQNAgent(None,state_size, action_size, False, '')

    solved_eps = 0
    steps_taken_for_completion = []
    episode_rewards = []

    my_actions = []
    for episode in range(400):
        actions = []

        episode_reward = []
        state = env.reset_action()
        print(state)
        print('episode {}/{}'.format(episode, 1))
        for step in range(60):


            action = agent.act(state)
            print("---"*2)
            print("Action: %s"% Action(action))
            print(state)

            next_state, reward, done = env.step_action(action)
            if Action(action)==Action.DOWN:
                done = True
            else:

                reward =-1
            print("Rew: %i",reward)
            print(next_state)

            episode_reward.append(reward)
            # total_reward = agent._predict(state)
            agent.memorize(state, action, reward, next_state, done)

            state = next_state

            actions.append(action)
            my_actions.append(action)


            if done:
                print("Solved - Total Steps: %i"%step)
                print(actions)
                break
            if len(agent.memory) > 10:
                agent.replay(5)

            break #### ALERTA

        episode_rewards.append(episode_reward)

    print(collections.Counter(my_actions))
    state = env.reset_action()
    total_reward = agent._predict(state)
    print(total_reward)
