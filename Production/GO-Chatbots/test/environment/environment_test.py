"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for testing the environment in the Goal-Oriented Dialogue Systems
"""

import os, sys, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core import constants as const
from core import util
from core.environment.environment import GOEnv
from core.dm.kb_helper import GOKBHelper
import cPickle as pickle


def test1_feasible_actions():
    """
    Method for preparing the feasible actions for the movie booking data set
    """

    # all of the request slots the agent can pick
    sys_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip',
                         'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain',
                         'price', 'actor', 'description', 'other', 'numberofkids']

    # all of the inform slots the agent can pick
    sys_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating',
                        'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor',
                        'description', 'other', 'numberofkids', 'taskcomplete', 'ticket']

    # initial feasible actions
    test_feasible_actions = [
        {'diaact': "confirm_question", 'inform_slots': {}, 'request_slots': {}},
        {'diaact': "confirm_answer", 'inform_slots': {}, 'request_slots': {}},
        {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}},
        {'diaact': "deny", 'inform_slots': {}, 'request_slots': {}},
    ]

    # aff the request slots
    for slot in sys_inform_slots:
        test_feasible_actions.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})

    # add the inform slots
    for slot in sys_request_slots:
        test_feasible_actions.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})

    return test_feasible_actions


def test1_environment():
    """
    Method for testing the Environment class for the movie booking data set
    """

    # the path to the act set and load it
    # TODO: change it with a relative path
    act_set_file_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/dia_acts.txt'
    act_set = util.text_to_dict(act_set_file_path)

    # the path to the slot set and load it
    # TODO: change it with a relative path
    slot_set_file_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/slot_set.txt'
    slot_set = util.text_to_dict(slot_set_file_path)

    # the path to the user goals and load it
    # TODO: change it with a relative path
    goal_set_file_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/user_goals_first_turn_template.part.movie.v1.p'
    goal_set = util.load_goal_set(goal_set_file_path)

    # the list of initial inform slots
    init_inform_slots = ['moviename']

    # the ultimate slot set
    ultimate_request_slot = 'ticket'

    # feasible actions
    test_feasible_actions = test1_feasible_actions()

    # define the arguments for the KB
    ultimate_request_slot = 'ticket'
    special_slots = ['numberofpeople']
    filter_slots = ['ticket', 'numberofpeople', 'taskcomplete', 'closing']
    knowledge_dict_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/movie_kb.1k.p'
    knowledge_dict = pickle.load(open(knowledge_dict_path, 'rb'))

    # Create the KB Helper class
    kb_helper = GOKBHelper(ultimate_request_slot, special_slots, filter_slots, knowledge_dict)

    # all params
    params = {}
    params[const.SIMULATION_MODE_KEY] = const.SEMANTIC_FRAME_SIMULATION_MODE
    params[const.IS_TRAINING_KEY] = True
    params[const.USER_TYPE_KEY] = const.RULE_BASED_USER
    params[const.STATE_TRACKER_TYPE_KEY] = const.RULE_BASED_STATE_TRACKER
    params[const.MAX_NB_TURNS] = 30
    params[const.SUCCESS_REWARD_KEY] = 2 * params[const.MAX_NB_TURNS]
    params[const.FAILURE_REWARD_KEY] = - params[const.MAX_NB_TURNS]
    params[const.PER_TURN_REWARD_KEY] = -1

    params[
        const.NLU_PATH_KEY] = "/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/models/nlu/lstm_[1468447442.91]_39_80_0.921.p"


    params[const.DIAACT_NL_PAIRS_PATH_KEY] = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/dia_act_nl_pairs.v6.json'
    params[
        const.NLG_PATH_KEY] = "/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p"

    env = GOEnv(act_set=act_set, slot_set=slot_set, goal_set=goal_set, init_inform_slots=init_inform_slots,
                ultimate_request_slot=ultimate_request_slot, feasible_actions=test_feasible_actions, kb_helper=kb_helper, params=params)

    env.reset()


    agt_action = {}
    agt_action[const.DIA_ACT_KEY] = 'request'

    agt_action[const.INFORM_SLOTS_KEY] = {}
    agt_action[const.INFORM_SLOTS_KEY]['theater'] = 'manville 12 plex'

    agt_action[const.REQUEST_SLOTS_KEY] = {}
    agt_action[const.REQUEST_SLOTS_KEY]['starttime'] = 'UKN'


    env.step(agt_action)


logging.basicConfig(filename='env_test.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Started')
test1_environment()
logging.info('Finished')
