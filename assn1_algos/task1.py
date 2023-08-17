"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

from ast import Num
from os import NGROUPS_MAX
import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def KL(p, q):
    if p==q:
        return 0
    if q==0 or q==1:
        return float("inf")
    if p==1:
        return np.log(1/q)
    if p==0:
        return -np.log(1-q)
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def get_kl_ucb(value, count, t, c=3):
    delta = 0.005
    target = (np.log(t) + c*np.log(np.log(t)))/count
    low, high = value, 1

    while high-low>=delta:
        mid = (low + high) / 2
        res = KL(value, mid)
        if target>res and target-res<=delta:
            return mid
        elif res > target:
            high = mid
        else:
            low = mid
    return low
# END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.values = np.zeros(num_arms)
        self.counts = np.ones(num_arms)
        self.ucb = np.zeros(num_arms)
        self.n = 1
        self.arms = num_arms
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(self.ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        if self.n != 1:
            for i in range(self.arms):
                new_ucb_value = get_kl_ucb(self.values[i], self.counts[i], self.n, c =3)
                self.ucb[i] = new_ucb_value
        
        self.n += 1
        # END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.values = np.zeros(num_arms)
        self.counts = np.ones(num_arms)
        self.ucb = np.zeros(num_arms)
        self.n = 1
        self.arms = num_arms
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        #if self.n == 1:
           # return np.random.randint(self.num_arms)
        #else:
        return np.argmax(self.ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        t = self.n
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        for i in range(self.arms):
            new_ucb_value = self.values[i] + np.sqrt(2*np.log(self.n) / self.counts[i])
            self.ucb[i] = new_ucb_value
        t+=1
        self.n = t
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.success = np.zeros(num_arms)
        self.failure = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.beta = np.full(num_arms, np.max(np.random.beta(1,1)))
        self.n = 1
        self.arms = num_arms
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        #if self.n == 1:
            #return np.random.randint(self.num_arms)
        #else:
        return np.argmax(self.beta)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.success[arm_index]
        new_value = value + reward
        self.success[arm_index] = new_value
        self.failure[arm_index] = n - new_value
        for i in range(self.arms):
            new_beta = np.max(np.random.beta(self.success[i]+1, self.failure[i]+1))
            self.beta[i] = new_beta
        
        self.n += 1       
        # END EDITING HERE
