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
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
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
        model.add(Dense(4, input_dim=self.state_size, activation='softmax'))
        model.add(Dense(10, activation='softmax'))
        model.add(Dense(4, ))

        model.compile(loss='mse', optimizer=Adam(lr=0.8))
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
            return np.argmax(act_values[0])
        return random.randrange(self.action_size)

    def replay(self, batch_size):

        # minibatch = random.sample(self.memory, batch_size)
        minibatch = self.memory  # fuerza a utilizarlas todas. Hay pocas combinaciones. Lo ideal es la anterior línea de código...

        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:

            # para todas las acciones vamos a corregir el reward, aprendemos de los fallos/aciertos.

            # partimos del estado inicial, y el que hemos obtenido con la acción.
            s = state.reshape((1, self.dim * self.dim))
            ns = next_state.reshape((1, self.dim * self.dim))
            reward_real = reward

            # si el juego no acabo!!! tendremos que mejorar mucho más !
            if not done:
                reward_real = (reward + self.gamma * np.amax(self.model.predict(ns)))  # la recompensa aprendida

            reward_real /= np.max(np.abs(reward_real), axis=0)  # la recompensa aprendida

            reward_real = np.reshape(reward_real, (1, self.action_size))
            # Ahora corregiremos los pesos. Vamos a hacer que converja más rápidamente.
            # Como los pesos, se generaron aleatoriamente... mejor si les damos un empujón.

            # Partimos de lo que el modelo dijo en su momento: el reward que calculo.
            # No tiene pq ser el valor del reward que dio, ahora dará otra solución pues el modelo ha ido evolucionando.
            # Lo tenemos que volver a calcular
            #reward_model = self.model.predict(s).reshape((self.dim, self.dim, self.action_size))
            reward_model = self.model.predict(s)

            ## TODO start
            for i in range(len(reward_real)):
                reward_model[i] = reward_real[i]
            #reward_model[0, action] = reward_real[0, action]

            # Te dejo este punto para ti, no quiero liarla con el shape del reward_real.
            # Necesitamos que el el reward_model sea igual a si mismo, pero asignando lo aprendido en el reward_real en solo aquel punto donde estaba el agente.
            # Es decir, si el agente ESTABA en la posición [X,Y], entonces el reward_model[X,Y]=reward_real[X,Y];
            ## TODO end. Quizás tengas que hacer un for, o un if...

            # El resto sigue igual. Guardamos el estado y su recompensa corregida.
            states.append(s)
            targets_f.append(reward_model)

        states = np.array(states).reshape(-1, self.dim * self.dim)
        targets_f = np.array(targets_f).reshape(-1, self.action_size)

        # Ahora volvemos a recalcular el modelo con lo que tenia que haber sido, con lo aprendido.
        history = self.model.fit(states, targets_f, epochs=1, verbose=0)
        #print("\n\n FIT PREDICT \n \n")

        #self.controller.predictActions()
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
