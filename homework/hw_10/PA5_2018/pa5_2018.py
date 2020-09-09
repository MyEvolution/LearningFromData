# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:05:14 2018

@author: XieTianwen
"""

import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
torch.set_default_dtype(torch.float64)
EPISODES = 1000
'''
some api you may use:
    torch.from_numpy()
    torch.view()
    torch.max()
    
'''


class net(nn.Module):         # build your net 
    def __init__(self, state_size, action_size):
        super(net, self).__init__()
        '''
        your code
        '''
        
        
    def forward(self, x):
        '''
        your code
        '''
        
        
        


class DQNAgent:         # bulid DQNagent
    def __init__(self, state_size, action_size, q_model, t_model):
        self.state_size = state_size            
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.q_model = q_model        # model
        self.t_model = t_model
        self.criterion = nn.MSELoss() # define loss
        self.optimiser = optim.Adam(self.q_model.parameters(),lr = 0.001) #define optimiser

    

    def remember(self, state, action, reward, next_state, done):     # save memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        '''
        your code
        '''
        # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        '''
        your codes,
        use data from memory to train you q_model
        '''
        
        if self.epsilon > self.epsilon_min:   # epsilon decay after each training
            self.epsilon *= self.epsilon_decay
        
        
    def update_t(self):    # update t_model weights
        torch.save(self.q_model.state_dict(), 'params.pkl')
        self.t_model.load_state_dict(torch.load('params.pkl'))


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    q_model = net(state_size, action_size)   # generate nets and DQNagent model
    t_model = net(state_size, action_size)
    agent = DQNAgent(state_size, action_size, q_model, t_model)
    done = False
    replace_target_iter =  25    
    batch_size = 100

    for e in range(EPISODES):
        state = env.reset()
        
        if e % replace_target_iter == 0:  # update t_model weights
            agent.update_t()
        for time in range(481):
            env.render()      # show the amination
            action = agent.act(state)     # chose action
            next_state, reward, done, _ = env.step(action) # Interact with Environment
            reward = reward if not done else -10  # get -10 reward if fail
            
            agent.remember(state, action, reward, next_state, done) # save memory
            state = next_state
            if done:                
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size and time % 10 == 0: # train q_model
                agent.replay(batch_size)
#                