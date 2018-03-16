"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the Goal-Oriented Dialogue Memory classes.
"""

import random

class GOMemory(object):
    """
    Class representing the agent memory in a Goal-Oriented Dialogue Systems.
    
    # Arguments:
        
        - ** experience_pool **: the pool of experiences. It is a list of tuples of a following form:
        
            - ** s_curr **: the current state the agent is perceiving
            - ** a_curr **: the action that agent took in s_curr
            - ** r_curr **: the reward that agent experienced after taking the action a_curr in s_curr
            - ** s_next **: the next state returned by the environment
            - ** done **: is the new state terminal or not
            
        - ** warmup_size **: the number of experience tuples to be saved during a warm-up
    """

    def __init__(self, warmup_size):
        self.experience_pool = []
        self.warmup_size = warmup_size

    def empty(self):
        """
        Method to empty the experience replay
        """

        self.experience_pool = []

    def append_warmup(self, s_curr, a_curr, r_curr, s_next, done):
        """
        Method to append an experience while the warm-up. It calls the `append` method.
        """

        if len(self.experience_pool) < self.warmup_size:
            self.append(s_curr, a_curr, r_curr, s_next, done)
            return True

        return False

    def append_simulation(self, s_curr, a_curr, r_curr, s_next, done):
        """
        Method to append an experience while epoch evaluation. It calls the `append` method.
        """

        self.append(s_curr, a_curr, r_curr, s_next, done)

    def append(self, s_curr, a_curr, r_curr, s_next, done):
        """
         Method to append new experience in the buffer
         
        # Arguments:
        
            - ** s_curr **: the current state the agent is perceiving
            - ** a_curr **: the action that agent took in s_curr
            - ** r_curr **: the reward that agent experienced after taking the action a_curr in s_curr
            - ** s_next **: the next state returned by the environment
            - ** done **: is the new state terminal or not 
        """

        new_experience = (s_curr, a_curr, r_curr, s_next, done)
        self.experience_pool.append(new_experience)

    def sample(self, batch_size):
        """
        Method to create a natch of randomly drawn experiences from the memory.
        
        # Arguments:
        
            - ** batch_size **: number of examples in one batch
            
        ** return **: batch of experiences
        """

        batch = [random.choice(self.experience_pool) for i in xrange(batch_size)]
        return batch

    def memory_size(self):
        return len(self.experience_pool)
