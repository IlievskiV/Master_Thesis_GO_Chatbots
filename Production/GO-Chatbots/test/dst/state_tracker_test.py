"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for testing the state trackers in the Goal-Oriented Dialogue Systems
"""

import os, sys, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core import constants as const
from core import util
from core.dst.state_tracker import GORuleBasedStateTracker
from core.dm.kb_helper import GOKBHelper
import cPickle as pickle

def test1_actions():
    """
    Method for preparing initial user action, the agent response and the second user action
    for the movie booking data set.
    """

    # initialize the initial user action
    init_usr_action = {}
    init_usr_action[const.DIA_ACT_KEY] = ''
    init_usr_action[const.INFORM_SLOTS_KEY] = {}
    init_usr_action[const.REQUEST_SLOTS_KEY] = {}
    init_usr_action[const.NL_KEY] = ''

    init_user_action_info_slot = 'moviename'
    init_user_action_info_slot_value = 'deadpool'
    init_usr_action_request_slot = 'ticket'

    # initialize the agent response
    agt_action = {}
    agt_action[const.DIA_ACT_KEY] = ''
    agt_action[const.INFORM_SLOTS_KEY] = {}
    agt_action[const.REQUEST_SLOTS_KEY] = {}
    agt_action[const.NL_KEY] = ''

    agt_action_request_slot = 'date'

    # initialize the user response
    usr_action = {}
    usr_action[const.DIA_ACT_KEY] = ''
    usr_action[const.INFORM_SLOTS_KEY] = {}
    usr_action[const.REQUEST_SLOTS_KEY] = {}
    usr_action[const.NL_KEY] = ''

    usr_action_inform_slot = 'date'
    usr_action_inform_slot_value = 'today'

    # build them
    init_usr_action[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY
    init_usr_action[const.INFORM_SLOTS_KEY][init_user_action_info_slot] = init_user_action_info_slot_value
    init_usr_action[const.REQUEST_SLOTS_KEY][init_usr_action_request_slot] = 'UKN'

    agt_action[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY
    agt_action[const.INFORM_SLOTS_KEY] = {}
    agt_action[const.REQUEST_SLOTS_KEY][agt_action_request_slot] = 'UKN'

    usr_action[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY
    usr_action[const.INFORM_SLOTS_KEY][usr_action_inform_slot] = usr_action_inform_slot_value
    usr_action[const.REQUEST_SLOTS_KEY] = {}

    return init_usr_action, agt_action, usr_action


def test1_rule_based_state_tracker():
    """
    Method for testing the rule-based state tracker on the movie booking data set
    """

    # the path to the act set and load it
    # TODO: change it with a relative path
    act_set_file_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/dia_acts.txt'
    act_set = util.text_to_dict(act_set_file_path)

    # the path to the slot set and load it
    # TODO: change it with a relative path
    slot_set_file_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/slot_set.txt'
    slot_set = util.text_to_dict(slot_set_file_path)

    # maximum number of turns
    max_nb_turns = 30

    # the dimensionality of the state
    act_cardinality = len(act_set)
    slot_cardinality = len(slot_set)
    state_dim = 2 * act_cardinality + 7 * slot_cardinality + 3 + max_nb_turns

    #  Create the KB Helper class
    ultimate_request_slot = 'ticket'
    special_slots = ['numberofpeople']
    filter_slots = ['ticket', 'numberofpeople', 'taskcomplete', 'closing']
    knowledge_dict_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/movie_kb.1k.p'
    knowledge_dict =pickle.load(open(knowledge_dict_path, 'rb'))

    kb_helper = GOKBHelper(ultimate_request_slot, special_slots, filter_slots, knowledge_dict)

    # create the state tracker
    state_tracker = GORuleBasedStateTracker(act_set=act_set, slot_set=slot_set, max_nb_turns=max_nb_turns, kb_helper=kb_helper)

    # reset the state tracker
    state_tracker.reset()

    # get three mock actions
    init_usr_action, agt_action, usr_action = test1_actions()

    state_tracker.update(init_usr_action, const.USR_SPEAKER_VAL)
    # produce state for the agent
    init_state = state_tracker.produce_state()

    state_tracker.update(agt_action, const.AGT_SPEAKER_VAL)
    state_tracker.update(usr_action, const.USR_SPEAKER_VAL)

    state = state_tracker.produce_state()



logging.basicConfig(filename='dst_test.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Started')
test1_rule_based_state_tracker()
logging.info('Finished')
