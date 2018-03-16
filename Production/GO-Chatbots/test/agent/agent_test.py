"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for testing the agents in the Goal-Oriented Dialogue Systems
"""

import os, sys, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core import constants as const
from core.agent.agents import GODQNAgent
from core.agent.processor import GOProcessor
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from keras.optimizers import Adam
from core import util
from keras.models import Sequential


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


def test1_dqn_agent():
    """
    Method for testing the dqn agent on the movie booking data set
    """

    # load the list of feasible actions
    test_feasible_actions = test1_feasible_actions()

    # `Agent` class arguments:
    processor = GOProcessor(test_feasible_actions)

    # `AbstractDQNAgent` class arguments:
    nb_actions = len(test_feasible_actions)
    memory = SequentialMemory(limit=1000000, window_length=4)
    gamma = .99
    batch_size = 32
    nb_steps_warmup = 1000
    train_interval = 4
    memory_interval = 1
    target_model_update = 10000

    # `DQNAgent` class arguments:
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    # `DQNAgent` class arguments:
    output_dim = nb_actions
    state_dimension = 256
    hidden_size = 80
    act_func = const.RELU

    # create the agent
    agent = GODQNAgent(processor=processor, nb_actions=nb_actions, memory=memory, gamma=gamma, batch_size=batch_size,
                       nb_steps_warmup=nb_steps_warmup, train_interval=train_interval, memory_interval=memory_interval,
                       target_model_update=target_model_update, policy=policy, output_dim=output_dim,
                       state_dimension=state_dimension, hidden_size=hidden_size, act_func=act_func)

    # compile the model
    agent.compile(Adam(lr=.00025), metrics=['mae'])


logging.basicConfig(filename='agent_test.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
logging.info('Started')
test1_dqn_agent()
logging.info('Finished')
