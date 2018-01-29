# -*- coding: utf-8 -*-

#A simple DQN learning using cartpoles!

import gym
import keras
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.995 #discount rate
        self.explor = 1.0 #exploration rate
        self.explor_min = 0.01
        self.explor_decay = 0.995
        self.learn_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        # Neural Net for Deep-Q
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size,activation='relu'))
#        model.add(Dropout(0.05))
#        model.add(Dense(50,activation='relu'))
#        model.add(Dropout(0.05))
        model.add(Dense(30,activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model
    
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
            target = reward #evaluate reward and use as update
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0]) #define loss function
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
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
    batch_size = 64

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
#            env.render() #see how the machines work
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.explor))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
#        if e % 10 == 0:
#            agent.save("./save/cartpole-dqn.h5")
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
