import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

import Constants


class DQNAgent:
    def __init__(self, state_size, action_size, use_existing_model, requested_model):
        self.state_size = state_size
        self.dim = int(np.sqrt(self.state_size))
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        if use_existing_model:
            self.load_weights(requested_model)
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(4, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(5, activation='relu'))
        #model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if state is not None:
            s = state.reshape((1, self.dim * self.dim))
            act_values = self.model.predict(s)
            #return np.argmax(act_values[0])  # returns action
            return np.argmax(act_values[0])
        return random.randrange(self.action_size)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        #minibatch = self.memory
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            if state is not None:
                s = state.reshape((1, self.dim * self.dim))
                sp = next_state.reshape((1, self.dim * self.dim))
                reward_real = reward
                if not done:
                    reward_real = (reward_real + self.gamma * np.amax(self.model.predict(sp)))  # reward_learned
                #print("REPLAY")
                #print(self.model.predict(s))
                #reward_model = self.model.predict(s).reshape((self.dim, self.dim, 2))
                #self.model.summary()

                #reward_model = self.model.predict(s).reshape((self.dim, self.dim))
                reward_model = self.model.predict(s)

                pos_action = 1

                # pos_action = get_services_position(state)
                #for row in range(len(reward_real)):
                #    for col in range(len(reward_real[row])):
                #        if row in pos_action:
                #            reward_model[row][col] = reward_real[row][col]
                #        else:
                #            reward_model[row][col] = [0., 0.]
                states.append(s)
                #targets_f.append(reward_model.reshape(1, self.dim * self.dim * 2))
                targets_f.append(reward_model)

        states = np.array(states).reshape(-1, self.dim * self.dim)
        targets_f = np.array(targets_f).reshape(-1, self.action_size)
        history = self.model.fit(states, targets_f, epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def _predict(self, state):
        reshaped_state = np.reshape(state, self.state_size)
        target = self.model.predict(reshaped_state, steps=10)
        size = self.state_size / 2
        reshaped_target = np.reshape(target, (size, size))
        return reshaped_target

    def save_model(self, model_name):
        self.model.save(model_name)

    def load_weights(self, requested_model):
        existing_model = tf.keras.models.load_model(requested_model, compile=False)
        self.model = self._build_model()
        self.model.set_weights(existing_model.get_weights())
