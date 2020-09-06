import numpy as np

class Agent(object):
    def __init__(self, policy, random_action_f, eps_max=1, eps_min=0.01, eps_decay=0.001):
        self.policy = policy
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.reset_explotation()
        self.random_action_f = random_action_f

    
    def act(self, state):
        self._update_epsilon()
        if np.random.rand() > self.epsilon:
            return self.policy(state)
        else:
            return self.random_action_f()

    def _update_epsilon(self):
        self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-1. * self.exploration_cnt * self.eps_decay)
        self.exploration_cnt += 1

    def reset_exploration(self):
        self.exploration_cnt = 0
        self.epsilon = self.eps_max