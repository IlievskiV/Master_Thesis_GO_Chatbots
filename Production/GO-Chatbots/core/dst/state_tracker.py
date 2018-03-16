"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>


A Python file for the Goal-Oriented Dialogue State Tracker classes.
"""

from core import constants as const

import numpy as np
import copy, logging, pprint
from core.dm.kb_helper import GOKBHelper


class GOStateTracker(object):
    """
    Abstract Base Class of all state trackers in the Goal-Oriented Dialogue Systems.
    
    # Class members:
    
        - ** history **: list of both user and agent actions, such that they are in alternating order
        - ** act_set **: the set of all intents used in the dialogue.
        - ** slot_set **: the set of all slots used in the dialogue.
        - ** act_set_cardinality **: the cardinality of the act set.
        - ** slot_set_cardinality **: the cardinality of the slot set.
        - ** current_slots **: a dictionary that keeps a running record of which slots are filled 
                        (inform slots) and which are requested (request slots)
        - ** state_dim **: the dimensionality of the state. It is calculated afterwards.
        - ** max_nb_turns **: the maximal number of dialogue turns
        - ** kb_helper **: the knowledge base helper class
    """

    def __init__(self, act_set=None, slot_set=None, max_nb_turns=None, kb_helper=None):
        """
        Constructor of the [GO State Tracker] class.
        """
        logging.info('Calling `GOStateTracker` constructor')
        self.pp = pprint.PrettyPrinter(indent=4)

        # the list of history
        self.history = []

        # The act and slot sets
        self.act_set = act_set
        self.slot_set = slot_set

        # The cardinality of the act and slot sets
        self.act_set_cardinality = len(self.act_set.keys())
        self.slot_set_cardinality = len(self.slot_set.keys())

        # initialize the running record of the slots
        self.current_slots = {}
        self.current_slots[const.INFORM_SLOTS_KEY] = {}
        self.current_slots[const.REQUEST_SLOTS_KEY] = {}
        self.current_slots[const.PROPOSED_SLOT_KEY] = {}
        self.current_slots[const.AGENT_REQUESTED_SLOT_KEY] = {}

        self.current_turn_nb = 0
        self.max_nb_turns = max_nb_turns + 4

        self.state_dim = 0

        # the knowledge base helper class
        self.kb_helper = kb_helper

    def __update_usr_action(self, usr_action):
        """
        Abstract private helper method to update the state tracker with the last user action.
        
        :param usr_action: the action user took
        :return: 
        """

        raise NotImplementedError()

    def __update_agt_action(self, agt_action):
        """
        Abstract private helper method to update the state tracker with the last agent action.
        
        :param agt_action: the action agent took
        :return: 
        """

        raise NotImplementedError()

    def get_history(self):
        """
        Getter method to get the dialogue history.
        
        :return: the history dialogue list 
        """
        return self.history

    def get_last_usr_action(self):
        """
        Getter method, to get the last user action, if any.
        
        :return: the last user action as dictionary, if any
        """
        return self.history[-1] if len(self.history) > 0 else None

    def get_last_agt_action(self):
        """
        Getter method, to get the last agent action, if any.
        
        :return: the last agent action as dictionary, if any
        """

        return self.history[-2] if len(self.history) > 1 else None

    def reset(self):
        """
        Abstract method for resetting the dialogue state tracker, usually at the beginning of a new episode.
        
        :return: true if the resetting was successful
        """

        raise NotImplementedError()

    def produce_state(self):
        """
        Abstract method to produce a representation for the current dialogue state.
        
        :return:
        """
        raise NotImplementedError()

    def update(self, action=None, speaker=""):
        """
        Abstract method to update the state tracker with the last user or agent action.
        
        :param action: user or agent action
        :param speaker: who took the action, the user or the agent
        :return: 
        """
        raise NotImplementedError()


class GORuleBasedStateTracker(GOStateTracker):
    """
    Class for Rule-Based state tracker in the Goal-Oriented Dialogue Systems.
    Extends the `GOStateTracker` class.
    
    # Class members:
        
        - ** state_dim **: the dimension of the state
    """

    def __init__(self, act_set=None, slot_set=None, max_nb_turns=None, kb_helper=None):
        """
        Constructor of the `GO Rule Based State Tracker` class.
        """
        logging.info('Calling `GORuleBasedStateTracker` constructor')

        super(GORuleBasedStateTracker, self).__init__(act_set, slot_set, max_nb_turns, kb_helper)

        self.act_cardinality = len(act_set)
        self.slot_cardinality = len(slot_set)
        self.state_dim = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_nb_turns


    def __encode_action_intent(self, action_intent):
        """
        Private helper method to create one-hot encoding for the intent of the current user or agent action.

        :param action_intent: string, describing the intent of the user or agent action
        :return: list in one-hot format
        """
        logging.info('Calling `GORuleBasedStateTracker` __encode_action_intent method')

        action_intent_encoding = np.zeros((1, self.act_set_cardinality))
        action_intent_encoding[0, self.act_set[action_intent]] = 1.0

        logging.debug("Action intent: '{0}'".format(self.pp.pformat(action_intent)))
        logging.debug("Encoding: '{0}'".format(self.pp.pformat(action_intent_encoding)))
        return action_intent_encoding

    def __encode_action_inform_slots(self, action_inform_slots):
        """
        Private helper method to create bag encoding for the inform slots in the current user or agent action.

        :param action_inform_slots: a dictionary of inform slots present in the current user or agent action
        :return: list in bag format
        """
        logging.info('Calling `GORuleBasedStateTracker` __encode_action_inform_slots method')

        action_inform_slots_encoding = np.zeros((1, self.slot_set_cardinality))
        for slot in action_inform_slots.keys():
            action_inform_slots_encoding[0, self.slot_set[slot]] = 1.0

        logging.debug("Action inform slots: '{0}'".format(self.pp.pformat(action_inform_slots)))
        logging.debug("Action inform slots encoding: '{0}'".format(self.pp.pformat(action_inform_slots_encoding)))

        return action_inform_slots_encoding

    def __encode_action_request_slots(self, action_request_slots):
        """
        Private helper method to create bag encoding for the request slots in the current user or agent action.

        :param action_request_slots: a dictionary of request slots in the current user or agent action
        :return: list in bag format
        """
        logging.info('Calling `GORuleBasedStateTracker` __encode_action_request_slot method')

        action_request_slots_encoding = np.zeros((1, self.slot_set_cardinality))
        for slot in action_request_slots.keys():
            action_request_slots_encoding[0, self.slot_set[slot]] = 1.0

        logging.debug("Action request slots: '{0}'".format(self.pp.pformat(action_request_slots)))
        logging.debug("Action request slots encoding: '{0}'".format(self.pp.pformat(action_request_slots_encoding)))

        return action_request_slots_encoding

    def __encode_all_inform_slots(self, all_inform_slots):
        """
        Private helper method to create bag encoding for all inform slots during the dialogue.

        :param all_inform_slots: a dictionary of all inform slots
        :return: list in bag format
        """
        logging.info('Calling `GORuleBasedStateTracker` __encode_all_inform_slots method')

        all_inform_slots_encoding = np.zeros((1, self.slot_set_cardinality))
        for slot in all_inform_slots:
            all_inform_slots_encoding[0, self.slot_set[slot]] = 1.0

        logging.debug("All inform slots: '{0}'".format(self.pp.pformat(all_inform_slots)))
        logging.debug("All inform slots encoding: '{0}'".format(self.pp.pformat(all_inform_slots_encoding)))

        return all_inform_slots_encoding

    def __encode_dialogue_turn_scaled(self, curr_turn_nb):
        """
        Private helper method for encoding the dialogue turn number scaled by 10

        :param curr_turn_nb: current dialogue turn number
        :return: one element list
        """
        logging.info('Calling `GORuleBasedStateTracker` __encode_dialogue_turn_scaled method')

        scaled_turn_encoding = np.zeros((1, 1)) + curr_turn_nb / 10.

        logging.debug("Current turn number: '{0}'".format(self.pp.pformat(curr_turn_nb)))
        logging.debug("Current scaled turn number encoding: '{0}'".format(self.pp.pformat(scaled_turn_encoding)))

        return scaled_turn_encoding

    def __encode_dialogue_turn(self, curr_turn_nb):
        """
        Private helper method to create one-hot encoding for the current dialogue turn

        :param curr_turn_nb: current dialogue turn number
        :return: list in one-hot format
        """
        logging.info('Calling `GORuleBasedStateTracker` __encode_dialogue_turn method')

        dialogue_turn_encoding = np.zeros((1, self.max_nb_turns))
        dialogue_turn_encoding[0, curr_turn_nb] = 1.0

        logging.debug("Current turn number: '{0}'".format(self.pp.pformat(curr_turn_nb)))
        logging.debug("Current one-hot turn number encoding: '{0}'".format(self.pp.pformat(dialogue_turn_encoding)))

        return dialogue_turn_encoding

    def __encode_kb_results_scaled(self, kb_results_dict):
        """
        Private helper method to create scaled counts encoding of the kb querying results

        :param kb_results_dict: dictionary of kb querying results
        :return: list of scaled kb querying results
        """
        logging.info('Calling `GORuleBasedStateTracker` __encode_kb_results_scaled method')

        kb_scaled_count_encoding = np.zeros((1, self.slot_set_cardinality + 1)) + kb_results_dict[
                                                                                      'matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_scaled_count_encoding[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.


        logging.debug("Knowledge-Base results: '{0}'".format(self.pp.pformat(kb_results_dict)))
        logging.debug("Scaled knowledge-Base results encoding: '{0}'".format(self.pp.pformat(kb_scaled_count_encoding)))

        return kb_scaled_count_encoding

    def __encode_kb_results_binary(self, kb_results_dict):
        """
        Private helper method to create binary encoding of the kb querying results.

        :param kb_results_dict: dictionary of kb querying results
        :return: 
        """

        logging.info('Calling `GORuleBasedStateTracker` __encode_kb_results_binary method')
        kb_binary_count_encoding = np.zeros((1, self.slot_set_cardinality + 1)) + np.sum(
            kb_results_dict[const.KB_MATCHING_ALL_CONSTRAINTS_KEY] > 0.)

        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_count_encoding[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)


        logging.debug("Knowledge-Base results: '{0}'".format(self.pp.pformat(kb_results_dict)))
        logging.debug("Binary knowledge-Base results encoding: '{0}'".format(self.pp.pformat(kb_binary_count_encoding)))

        return kb_binary_count_encoding

    def __update_usr_action(self, usr_action):
        """
        Abstract method implementation.
        """

        logging.info('Calling `GORuleBasedStateTracker` __update_usr_action method')
        # Iterate over the inform slots from the last user action and update the state tracker running record
        for slot in usr_action[const.INFORM_SLOTS_KEY].keys():
            self.current_slots[const.INFORM_SLOTS_KEY][slot] = usr_action[const.INFORM_SLOTS_KEY][slot]
            # if the current inform slot was in the requested slots in the past, delete it
            if slot in self.current_slots[const.REQUEST_SLOTS_KEY].keys():
                del self.current_slots[const.REQUEST_SLOTS_KEY][slot]

        # Iterate over the request slots from the last user action and update the state tracker running record
        for slot in usr_action[const.REQUEST_SLOTS_KEY].keys():
            if slot not in self.current_slots[const.REQUEST_SLOTS_KEY].keys():
                self.current_slots[const.REQUEST_SLOTS_KEY][slot] = const.UNKNOWN_SLOT_VALUE

        # Produce a record for the history and add the last user action in the history
        new_history_record = {}
        new_history_record[const.TURN_NB_KEY] = self.current_turn_nb
        new_history_record[const.SPEAKER_TYPE_KEY] = const.USR_SPEAKER_VAL
        new_history_record[const.DIA_ACT_KEY] = usr_action[const.DIA_ACT_KEY]
        new_history_record[const.INFORM_SLOTS_KEY] = usr_action[const.INFORM_SLOTS_KEY]
        new_history_record[const.REQUEST_SLOTS_KEY] = usr_action[const.REQUEST_SLOTS_KEY]

        self.history.append(copy.deepcopy(new_history_record))

        # increase the turn number for one
        self.current_turn_nb += 1

        return True

    def __update_agt_action(self, agt_action):
        """
        Abstract method implementation.
        """

        logging.info('Calling `GORuleBasedStateTracker` __update_agt_action method')
        # Make a copy and call KB helper methods to fill in the values for the inform slots

        agt_action_copy = copy.deepcopy(agt_action)
        inform_slots_from_kb = self.kb_helper.fill_inform_slots(agt_action_copy[const.INFORM_SLOTS_KEY],
                                                                self.current_slots)

        # Iterate over the inform slots from the KB and update the state tracker running record
        for slot in inform_slots_from_kb.keys():
            self.current_slots[const.PROPOSED_SLOT_KEY][slot] = inform_slots_from_kb[slot]
            self.current_slots[const.INFORM_SLOTS_KEY][slot] = inform_slots_from_kb[slot]
            # if the current inform slot was in the requested slots in the past, delete it
            if slot in self.current_slots[const.REQUEST_SLOTS_KEY].keys():
                del self.current_slots[const.REQUEST_SLOTS_KEY][slot]

        # Iterate over the request slots from the last agent action and update the state tracker running record
        for slot in agt_action_copy[const.REQUEST_SLOTS_KEY].keys():
            if slot not in self.current_slots[const.AGENT_REQUESTED_SLOT_KEY].keys():
                self.current_slots[const.AGENT_REQUESTED_SLOT_KEY][slot] = const.UNKNOWN_SLOT_VALUE

        # Produce a record for the history and add the last agent action in the history
        new_history_record = {}
        new_history_record[const.TURN_NB_KEY] = self.current_turn_nb
        new_history_record[const.SPEAKER_TYPE_KEY] = const.AGT_SPEAKER_VAL
        new_history_record[const.DIA_ACT_KEY] = agt_action_copy[const.DIA_ACT_KEY]
        new_history_record[const.INFORM_SLOTS_KEY] = agt_action_copy[const.INFORM_SLOTS_KEY]
        new_history_record[const.REQUEST_SLOTS_KEY] = agt_action_copy[const.REQUEST_SLOTS_KEY]

        self.history.append(copy.deepcopy(new_history_record))

        # increase the turn number for one
        self.current_turn_nb += 1

        return True

    def get_state_dimension(self):
        """
        
        :return: 
        """

        return self.state_dim

    def reset(self):
        """
        Method to reset the rule-based dialogue state tracker. Overrides the super class method.
        
        :return: true if the resetting was successful, false otherwise
        """

        logging.info('Calling `GORuleBasedStateTracker` reset method')
        # clear the history
        self.history = []

        # clear the running record of filled slots
        self.current_slots = {}
        self.current_slots[const.INFORM_SLOTS_KEY] = {}
        self.current_slots[const.REQUEST_SLOTS_KEY] = {}
        self.current_slots[const.PROPOSED_SLOT_KEY] = {}
        self.current_slots[const.AGENT_REQUESTED_SLOT_KEY] = {}

        # set turn number to 0
        self.current_turn_nb = 0

        return True

    def produce_state(self):
        """
        Abstract method implementation.
        Method to produce a representation for the current dialogue state. In this rule-based state tracker it includes:
        
            - one-hot encoding of the last user and the agent action intent
            - bag encoding of the last user and the agent action inform slots
            - bag encodi=ng of the last user and the agent action request slots
            - bag encoding of all inform slots in the dialogue so far
            - dialogue turn number scaled by 10
            - one-hot encoding of the dialogue turn number
            - kb querying results scaled by 100
            - kb querying results in a binary form, like present not present
        
        :return: list of numbers representing the current state
        """

        logging.info('Calling `GORuleBasedStateTracker` produce_state method')

        # get the last user and agent action
        last_usr_action = self.get_last_usr_action()
        last_agt_action = self.get_last_agt_action()

        if last_usr_action:
            # user action intent encoding
            usr_action_intent_encoding = self.__encode_action_intent(last_usr_action[const.DIA_ACT_KEY])

            # user inform slots encoding
            usr_action_inform_slots_encoding = self.__encode_action_inform_slots(
                last_usr_action[const.INFORM_SLOTS_KEY])

            # user request slots encoding
            usr_action_request_slots_encoding = self.__encode_action_request_slots(
                last_usr_action[const.REQUEST_SLOTS_KEY])
        else:
            # user action intent encoding
            usr_action_intent_encoding = np.zeros((1, self.act_set_cardinality))

            # user inform slots encoding
            usr_action_inform_slots_encoding = np.zeros((1, self.slot_set_cardinality))

            # user request slots encoding
            usr_action_request_slots_encoding = np.zeros((1, self.slot_set_cardinality))

        if last_agt_action:
            # agent action intent encoding
            agt_action_intent_encoding = self.__encode_action_intent(last_agt_action[const.DIA_ACT_KEY])

            # agent inform slots encoding
            agt_action_inform_slots_encoding = self.__encode_action_inform_slots(
                last_agt_action[const.INFORM_SLOTS_KEY])

            # agent request slots encoding
            agt_action_request_slots_encoding = self.__encode_action_request_slots(
                last_agt_action[const.REQUEST_SLOTS_KEY])
        else:
            # agent action intent encoding
            agt_action_intent_encoding = np.zeros((1, self.act_set_cardinality))

            # agent inform slots encoding
            agt_action_inform_slots_encoding = np.zeros((1, self.slot_set_cardinality))

            # agent request slots encoding
            agt_action_request_slots_encoding = np.zeros((1, self.slot_set_cardinality))

        # all inform slots in the dialogue so far
        all_inform_slots_encoding = self.__encode_all_inform_slots(self.current_slots[const.INFORM_SLOTS_KEY])

        # scaled dialogue turn number encoding
        scaled_turn_encoding = self.__encode_dialogue_turn_scaled(self.current_turn_nb)

        # one-hot dialogue turn number encoding
        dialogue_turn_encoding = self.__encode_dialogue_turn(self.current_turn_nb)

        # TODO: create the KB helper class to query the KB
        kb_results_dict = self.kb_helper.database_results_for_agent(self.current_slots)

        # kb scaled encoding
        kb_scaled_count_encoding = self.__encode_kb_results_scaled(kb_results_dict)

        # kb binary encoding
        kb_binary_count_encoding = self.__encode_kb_results_binary(kb_results_dict)

        # stack everything in one vector
        final_representation = np.hstack(
            [usr_action_intent_encoding, usr_action_inform_slots_encoding, usr_action_request_slots_encoding,
             agt_action_intent_encoding, agt_action_inform_slots_encoding, agt_action_request_slots_encoding,
             all_inform_slots_encoding, scaled_turn_encoding, dialogue_turn_encoding, kb_binary_count_encoding,
             kb_scaled_count_encoding])[0]

        logging.debug("State: '{0}'".format(self.pp.pformat(final_representation)))
        return np.array(final_representation)[np.newaxis]

    def update(self, action=None, speaker=None):
        """
        Abstract method implementation
        """

        logging.info('Calling `GORuleBasedStateTracker` update method')
        # the function should be called proplerly
        assert (action and speaker)



        if speaker == const.USR_SPEAKER_VAL:
            return self.__update_usr_action(action)
        else:
            return self.__update_agt_action(action)


class GOModelBasedStateTracker(GOStateTracker):
    """
    Class for Model-Based state tracker in the Goal-Oriented Dialogue Systems.
    Extends the `GOStateTracker` class.
    
    # Class members:
    
        - ** model_path **: the path to save or load the model
    """

    def __init__(self, act_set=None, slot_set=None, max_nb_turns=None, kb_helper=None, model_path=None):
        super(GOModelBasedStateTracker, self).__init__(act_set, slot_set, max_nb_turns, kb_helper)

        logging.info('Calling `GOModelBasedStateTracker` constructor')

        self.state_dim = 0 # TODO: calculate when implemented
        self.model_path = model_path

    def __update_usr_action(self, usr_action):
        # TODO
        raise NotImplementedError()

    def __update_agt_action(self, agt_action):
        # TODO
        raise NotImplementedError()

    def reset(self):
        """
        Method to reset the model-based dialogue state tracker. Overrides the super class method.

        :return: true if the resetting was successful, false otherwise
        """

        # TODO
        raise NotImplementedError()

    def produce_state(self):
        # TODO
        raise NotImplementedError()

    def update(self, action=None, speaker=""):
        # TODO
        raise NotImplementedError()
