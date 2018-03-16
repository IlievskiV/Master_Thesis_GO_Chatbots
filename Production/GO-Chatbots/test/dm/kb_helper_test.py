"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for testing the knowledge base in the Goal-Oriented Dialogue Systems
"""

import os, sys, logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core import constants as const
from core import util
from core.dm.kb_helper import GOKBHelper
import cPickle as pickle

def test1_kb_helper():
    """
    Method for testing the KB Helper for the movie booking
    """

    # define the arguments for the
    ultimate_request_slot = 'ticket'
    special_slots = ['numberofpeople']
    filter_slots = ['ticket', 'numberofpeople', 'taskcomplete', 'closing']
    knowledge_dict_path = '/Users/vladimirilievski/Desktop/Vladimir/Master_Thesis_Swisscom/GitHub Repo/GO-Chatbots/resources/data/movie_kb.1k.p'

    # Create the KB Helper class
    knowledge_dict = pickle.load(open(knowledge_dict_path, 'rb'))
    kb_helper = GOKBHelper(ultimate_request_slot, special_slots, filter_slots, knowledge_dict)

    current_slots = {}
    current_slots[const.INFORM_SLOTS_KEY] = {}
    current_slots[const.INFORM_SLOTS_KEY]['moviename'] = 'deadpool'
    current_slots[const.INFORM_SLOTS_KEY]['date'] = 'today'
    current_slots[const.INFORM_SLOTS_KEY]['numberofpeople'] = '2'

    database_results = kb_helper.database_results_for_agent(current_slots)




logging.basicConfig(filename='kb_helper_test.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
logging.info('Started')
test1_kb_helper()
logging.info('Finished')