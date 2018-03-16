"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>
"""

import os, sys, logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.dm.dialogue_system import GODialogSys
from core import constants as const
from core import util
from core.agent.policy import GORuleBasedPolicy
from experiments.draw_learning_curve import load_performance_file, draw_learning_curve

from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from core.agent.memory import GOMemory


def feasible_actions():
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
    agt_feasible_actions = [
        {'diaact': "confirm_question", 'inform_slots': {}, 'request_slots': {}},
        {'diaact': "confirm_answer", 'inform_slots': {}, 'request_slots': {}},
        {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}},
        {'diaact': "deny", 'inform_slots': {}, 'request_slots': {}},
    ]

    # aff the request slots
    for slot in sys_inform_slots:
        agt_feasible_actions.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})

    # add the inform slots
    for slot in sys_request_slots:
        agt_feasible_actions.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})

    return agt_feasible_actions


def prepare_dialogue_system():
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

    kb_special_slots = ['numberofpeople']
    kb_filter_slots = ['ticket', 'numberofpeople', 'taskcomplete', 'closing']

    # feasible actions
    agt_feasible_actions = feasible_actions()

    # the agent memory
    agt_memory = GOMemory(warmup_size=1000)

    # the agent policy
    agt_policy = EpsGreedyQPolicy(eps=0.1)

    warmup_policy_request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
    agt_warmup_policy = GORuleBasedPolicy(feasible_actions=agt_feasible_actions, request_set=warmup_policy_request_set)

    # testing policy
    agt_eval_policy = EpsGreedyQPolicy(eps=0.1)

    # all system params
    params = {}
    params[const.MAX_NB_TURNS] = 40
    params[
        const.KB_PATH_KEY] = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/movie_kb.1k.p'

    # Environment params
    params[const.SIMULATION_MODE_KEY] = const.SEMANTIC_FRAME_SIMULATION_MODE
    params[const.IS_TRAINING_KEY] = True
    params[const.USER_TYPE_KEY] = const.RULE_BASED_USER
    params[const.STATE_TRACKER_TYPE_KEY] = const.RULE_BASED_STATE_TRACKER
    params[const.SUCCESS_REWARD_KEY] = 2 * params[const.MAX_NB_TURNS]
    params[const.FAILURE_REWARD_KEY] = - params[const.MAX_NB_TURNS]
    params[const.PER_TURN_REWARD_KEY] = -1

    # TODO: change it with a relative path
    params[
        const.NLU_PATH_KEY] = "/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/models/nlu/lstm_[1468447442.91]_39_80_0.921.p"

    params[
        const.DIAACT_NL_PAIRS_PATH_KEY] = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/dia_act_nl_pairs.v6.json'
    params[
        const.NLG_PATH_KEY] = "/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p"

    # Agent params
    params[const.AGENT_TYPE_KEY] = const.AGENT_TYPE_DQN
    params[const.GAMMA_KEY] = .99
    params[const.BATCH_SIZE_KEY] = 32
    params[const.NB_STEPS_WARMUP_KEY] = 10000
    params[const.TRAIN_INTERVAL_KEY] = 20
    params[const.MEMORY_INTERVAL_KEY] = 50
    params[const.TARGET_MODEL_UPDATE_KEY] = 100
    params[const.ENABLE_DOUBLE_DQN_KEY] = False
    params[const.ENABLE_DUELING_NETWORK_KEY] = False
    params[const.DUELING_TYPE_KEY] = 'avg'
    params[const.HIDDEN_SIZE_KEY] = 80
    params[const.ACTIVATION_FUNCTION_KEY] = const.RELU

    # create the dialogue system
    dialogue_sys = GODialogSys(act_set=act_set, slot_set=slot_set, goal_set=goal_set,
                               init_inform_slots=init_inform_slots, ultimate_request_slot=ultimate_request_slot,
                               kb_special_slots=kb_special_slots, kb_filter_slots=kb_filter_slots,
                               agt_feasible_actions=agt_feasible_actions, agt_memory=agt_memory, agt_policy=agt_policy,
                               agt_warmup_policy=agt_warmup_policy,
                               agt_eval_policy=agt_eval_policy, params=params)

    return dialogue_sys


dialogue_system = prepare_dialogue_system()

nb_epochs = 500
nb_warmup_episodes = 120
nb_episodes_per_epoch = 100
res_path = "performance.json"
weights_file_name = "exp1_weights.h5f"


logging.basicConfig(filename='experiment1.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
logging.info('Started')

dialogue_system.train(nb_epochs=nb_epochs, nb_warmup_episodes=nb_warmup_episodes,
                      nb_episodes_per_epoch=nb_episodes_per_epoch,
                      res_path=res_path, weights_file_name=weights_file_name)

logging.info('Finished')

numbers = load_performance_file(res_path)
# draw_learning_curve(numbers)
