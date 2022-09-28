
import numpy as np
import pandas as pd

class FILTERS:
  def __init__(self, X, filtersno):
    self.filtersno = filtersno
    self.X = X
    self.states = self.filtersno * [None]
    self.states[0] = self.X
    self.moves = []
    
  def action_hist(self, action):
    self.moves.append(action)
    return self.moves

  def reset_acts(self):
    self.moves = []
    self.states = self.filtersno * [None]
    self.states[0] = self.X
    return self.moves, self.states
  
  def next_states(self, S, A):
    R=0
    self.action_hist(A)
    SS = S+1
    if S!=self.filtersno-2:
      if A=='active':
        if self.moves[len(self.moves)-2] == 'active':
          self.states[SS] = self.states[S]/3
        else:
          self.states[SS] = self.states[S]/2
      else:
        self.states[SS] = self.states[S]
    else:
      self.states[SS]=self.states[S]
      if self.states[SS]>=0.5 and self.states[SS]<1:
        R=1
    return SS, R

