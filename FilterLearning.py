
import numpy as np
import pandas as pd
import random
import pickle
import filters
import os

class Agent:
  def __init__(self, X, gamma = 0.9, alpha = 0.1, epsilon = 0.2, episodes = 15, nfilters = 7):
    self.actions = ['active', 'deactivate']
    self.X = X
    self.nfilters = nfilters
    self.filter = filters.FILTERS(self.X, self.nfilters)
    self.q_table = pd.DataFrame(np.zeros((self.nfilters, len(self.actions))), columns=self.actions)
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.episodes = episodes
    self.brain = dict()

  def check_q(self):
    if not os.path.isfile("brain"):
      with open("brain",'wb') as q_file:
        pickle.dump(self.brain, q_file)
    else:
      with open('brain', 'rb') as q_file:
        q = pickle.load(q_file)
        self.brain = q
    return q_file


  def rl(self):
    q_file = self.check_q()
    for eps in range(self.episodes):
      step_counter = 0
      filter_num = 0
      is_terminal = False
      #update_env(S, eps, step_counter)
      while not is_terminal:
        if (np.random.uniform() > self.epsilon) or (self.q_table.loc[filter_num, :]==0).all():
          action = np.random.choice(self.actions)
        else:
          action = self.q_table.loc[filter_num, :].idxmax()
        next_num, R = self.filter.next_states(filter_num, action)
        state = self.filter.states
        mov = self.filter.moves
        q_predict = self.q_table.loc[filter_num, action]
        if next_num != self.nfilters-1:
          q_target = R + self.gamma * self.q_table.loc[next_num, :].max()
        else:
          q_target = R
          self.filter.reset_acts()
          is_terminal = True
        self.q_table.loc[filter_num, action] += self.alpha * (q_target - q_predict)
        filter_num = next_num
        #update_env(S, eps, step_counter+1)
      print("Learning episode %2d is completed. " % (eps))
      print(" ")
      print("Ozone levels of filters: ", state)
      print("Actions of filters: ", mov)
      self.brain[self.X] = self.q_table
    try:
      with open("brain", 'rb') as q_file:
        self.brain.update(pickle.load(q_file))
      with open("brain", 'wb') as q_file:
        pickle.dump(self.brain, q_file)
    except:
      print("Q table is could not be created for filter actions")
    return self.q_table, q_file

