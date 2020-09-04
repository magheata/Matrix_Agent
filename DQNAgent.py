from __future__ import division
import os
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorflow as tf


class DQNAgent:
    def __init__(self, controller, state_size, action_size, use_existing_model, requested_model):
        self.controller = controller
        self.state_size = state_size
        self.dim = int(np.sqrt(self.state_size))
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.alpha_decay = 0.1
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.requested_model = requested_model
        if use_existing_model:
            self.load_weights("model/{}/{}".format(requested_model, requested_model))
        self.update_target_model()

    def set_requested_model(self, requested_model):
        self.requested_model = requested_model

    def setStateSize(self, state_size):
        self.state_size = state_size
        self.dim = int(np.sqrt(self.state_size))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(4, input_dim=self.state_size, activation='linear'))
        model.add(Dense(25, activation='linear'))
        model.add(Dense(25, activation='linear'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.alpha_decay))

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        s = state.reshape((1, self.dim * self.dim))
        act_values = self.model.predict(s)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        states, targets_f = [], []
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            s = state.reshape((1, self.dim * self.dim))
            ns = next_state.reshape((1, self.dim * self.dim))
            value_reward = reward
            if not done:
                value_reward = (value_reward + self.gamma * np.amax(self.model.predict(ns)))  # la recompensa aprendida
            target = self.model.predict(s)
            target[0, action] = value_reward
            states.append(s)
            targets_f.append(target)
        states = np.array(states).reshape(-1, self.dim * self.dim)
        targets_f = np.array(targets_f).reshape(-1, self.action_size)
        history = self.model.fit(states, targets_f, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def _predict(self, state):
        reshaped_state = np.reshape(state, (1, self.state_size))
        target = self.model.predict(reshaped_state)
        return target

    def save_model(self, model_name):
        parent_directory = os.getcwd()
        model_directory = "model"
        path = os.path.join(parent_directory, model_directory)

        if not os.path.isdir(os.path.join(path, model_name)):
            os.mkdir(os.path.join(path, model_name))

        self.requested_model = model_name
        self.model.save("model/{}/{}".format(model_name, model_name), model_name, True)

    def load_weights(self, requested_model):
        existing_model = tf.keras.models.load_model(requested_model, compile=False)
        self.model = self._build_model()
        self.model.set_weights(existing_model.get_weights())
