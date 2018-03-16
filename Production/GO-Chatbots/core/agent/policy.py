"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the Goal-Oriented Dialogue Policy classes.
"""

from rl.policy import Policy
from core import constants as const
import numpy as np

class GORuleBasedPolicy(Policy):
    """
    Class for the Goal-Oriented Rule Based Policy. With a prob. epsilon it takes a random action,
    otherwise it selects rule-based action.
    
    # Arguments:
    
        ** eps **:
        ** feasible_actions **:
        ** request_set **:
        
        
    """

    def __init__(self, eps=.1, feasible_actions=None, request_set=None, *args, **kwargs):
        super(GORuleBasedPolicy, self).__init__(*args, **kwargs)


        self.eps = eps

        self.feasible_actions = feasible_actions
        self.request_set = request_set

        self.current_slot_id = 0
        self.phase = 0

    def __get_action_index(self, agt_action):
        """
        Private helper method to convert the action to index
        """

        for (i, action) in enumerate(self.feasible_actions):
            if agt_action == action:
                return i

        raise Exception("The agent response does not exist")

    def __select_random_action(self):
        """
        Private helper method for selecting a random action.
        """

        nb_actions = len(self.feasible_actions)
        action_idx = np.random.random_integers(0, nb_actions-1)

        return action_idx



    def __select_rule_based_action(self):
        """
        Private helper method for selecting a rule based action 
        """

        agt_action = {}

        # if it was not asked for some request slot
        if self.current_slot_id < len(self.request_set):
            # take the next slot
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            # make the action
            agt_action[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY
            agt_action[const.INFORM_SLOTS_KEY] = {}
            agt_action[const.REQUEST_SLOTS_KEY] = {slot: "UNK"}

        # if it was asked every request slot
        elif self.phase == 0:

            # make an inform action with task complete slot
            agt_action[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY
            agt_action[const.INFORM_SLOTS_KEY] = {const.TASK_COMPLETE_SLOT : "PLACEHOLDER"}
            agt_action[const.REQUEST_SLOTS_KEY] = {}

            self.phase += 1
        # otherwise, just reply thanks
        elif self.phase == 1:

            agt_action[const.DIA_ACT_KEY] = const.THANKS_DIA_ACT_KEY
            agt_action[const.INFORM_SLOTS_KEY] = {}
            agt_action[const.REQUEST_SLOTS_KEY] = {}

        action_idx = self.__get_action_index(agt_action)
        return action_idx

    def reset(self):
        """
        Method to reset the policy
        """

        self.current_slot_id = 0
        self.phase = 0

    def select_action(self, **kwargs):
        """A method to select an action"""

        if np.random.uniform() < self.eps:
            action_idx = self.__select_random_action()
        else:
            action_idx = self.__select_rule_based_action()

        return action_idx

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def get_config(self):
        return {}
