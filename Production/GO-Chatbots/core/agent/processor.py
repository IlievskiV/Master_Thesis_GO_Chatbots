"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the GO Dialogue System Processor classes
"""

from rl.core import Processor
import copy, logging

class GOProcessor(Processor):
    """
    Class for the Goal-Oriented Processor which the mediator between the agent and the environment.
    It is processing the observation and the reward from the environment, in order to be in a correct
    format for the agent.
    
    # Arguments:
    
        - ** feasible_actions **: all feasible actions the agent might take
    """

    def __init__(self, feasible_actions=None, *args, **kwargs):
        """
         Constructor of the [GOProcessor] class
        """
        super(GOProcessor, self).__init__(*args, **kwargs)

        logging.info('Calling `GOProcessor` constructor')
        self.feasible_actions = feasible_actions

    def process_observation(self, observation):
        """
        Method for processing an observation from the environment. Overrides the super class method.
        
        :param observation: the observation from the environment to be processed
        :return: processed observation
        """

        # TODO: think about possible changes
        logging.info('Calling `GOProcessor` process_observation method')
        return observation

    def process_reward(self, reward):
        """
        Method for processing a reward from the environment. Overrides the super class method.
        
        :param reward: the reward from the environment to be processed
        :return: processed observation
        """

        # TODO: think about possible changes
        logging.info('Calling `GOProcessor` process_reward method')
        return reward

    def process_info(self, info):
        """
        Method for processing the info from the environment. Overrides the super class method.
        
        :param info: the info from the environment to be processed
        :return: processed info
        """

        # TODO: think about possible changes
        logging.info('Calling `GOProcessor` process_info method')
        return info

    def process_action(self, action):
        """
        Method for processing the agent action, in order to be suitable for the environment.
        Overrides the super class method.
        
        :param action: the agent action provided as a number
        :return: corresponding agent action as a dialogue act
        """

        logging.info('Calling `GOProcessor` process_action method')
        return copy.deepcopy(self.feasible_actions[action])

    def process_state_batch(self, batch):
        """
        Method for processing an entire batch of observations. Overrides the super class method.
        
        :param batch: the batch of observations to be processed
        :return: the processed batch
        """

        # TODO: think about possible changes
        logging.info('Calling `GOProcessor` process_state_batch method')
        return batch
