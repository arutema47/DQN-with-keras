# -*- coding: utf-8 -*-

#with better loss functions

import gym
import keras
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
from keras import backend as K

EPISODES = 2000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        self.gamma = 0.95 #discount rate
        self.explor = 1.0 #exploration rate
        self.explor_min = 0.01
        self.explor_decay = 0.99
        self.learn_rate = 0.001
        self.model = self._build_model()
        self.target_model=self._build_model()
        self.update_target_model()
        
    def _huber_loss(self, target, prediction):
        #sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

        
    def _build_model(self):
        # Neural Net for Deep-Q
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size,activation='relu'))
#        model.add(Dropout(0.05))
#        model.add(Dense(50,activation='relu'))
#        model.add(Dropout(0.05))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learn_rate))
        return model
    
    def update_target_model(self):
        #copy weights from model to target model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <=self.explor:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) #returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            #target = reward #evaluate reward and use as update
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)] #define loss function
#            target_f = self.model.predict(state)
#            target_f[0][action] = target
            self.model.fit(state, target, epochs=1, verbose=0)
            #whats target_f??
        if self.explor > self.explor_min:
            self.explor *=self.explor_decay
            
            
## training
if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size,action_size)
    # Iterate the game
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(2000):
            env.render() #see how the machines work
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.explor))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
#        if e % 10 == 0:
#            agent.save("./save/cartpole-dqn.h5")

    #Play it!!
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size,action_size)
    # Iterate the game
    done = False
    batch_size = 32
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(2000):
        env.render() #see how the machines work
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
