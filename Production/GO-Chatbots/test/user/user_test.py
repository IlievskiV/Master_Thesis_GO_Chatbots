"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for testing the users in the Goal-Oriented Dialogue Systems
"""
import os, sys, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core import constants as const
from core.user.users import GORuleBasedUser
from core import util




def test1_rule_based_user():
    """
    Method for testing the rule-based user on the movie booking data set
    """
    simulation_mode = const.SEMANTIC_FRAME_SIMULATION_MODE

    # the path to the user goals and load it
    # TODO: change it with a relative path
    goal_set_file_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/user_goals_first_turn_template.part.movie.v1.p'
    goal_set = util.load_goal_set(goal_set_file_path)

    # the path to the act set and load it
    # TODO: change it with a relative path
    act_set_file_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/dia_acts.txt'
    act_set = util.text_to_dict(act_set_file_path)

    # the path to the slot set and load it
    # TODO: change it with a relative path
    slot_set_file_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/slot_set.txt'
    slot_set = util.text_to_dict(slot_set_file_path)

    # the list of initial inform slots
    init_inform_slots = ['moviename']

    # the ultimate slot set
    ultimate_request_slot = 'ticket'

    # create the rule-based user
    user = GORuleBasedUser(simulation_mode, goal_set, act_set, init_inform_slots, ultimate_request_slot)

    # reset the user
    user.reset()

logging.basicConfig(filename='user_test.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
logging.info('Started')
test1_rule_based_user()
logging.info('Finished')