"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the helper Knowledge Base class
"""

from collections import defaultdict
from core import constants as const
import logging
import cPickle as pickle
import pprint

class GOKBHelper(object):
    """
    Helper class for the agent to query the provided knowledge base. It provides methods for querying and filling
    the slot values based on the results.
    
    #Arguments:
        
        - ** ultimate_request_slot **: the slot that is the actual goal of the user, and everything is around this slot.
        - ** special_slots **: list of special slots to check, if any
        - ** filter_slots **: the list of slots to filter out
        - ** knowledge_dict **: the dictionary to help query the knowledge base
        - ** cached_kb **:
        - ** cached_kb_slot **:
    
    """

    def __init__(self, ultimate_request_slot = None, special_slots = None, filter_slots = None, knowledge_dict = None):
        """Constructor of the `GOKBHelper` class"""
        logging.info('Calling `GOKBHelper` constructor ')
        self.pp = pprint.PrettyPrinter(indent=4)

        self.ultimate_request_slot = ultimate_request_slot
        self.special_slots = special_slots
        self.filter_slots = filter_slots

        # load the knowledge dictionary
        self.knowledge_dict = knowledge_dict
        logging.debug("Knowledge dictionary: '{0}'".format(self.pp.pformat(self.knowledge_dict)))

        self.cached_kb = defaultdict(list)
        self.cached_kb_slot = defaultdict(list)

    def fill_inform_slots(self, inform_slots_to_be_filled, current_slots):
        """
        Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)
        
        # Arguments
        
            - ** inform_slots_to_be_filled **:  the slots for which the agent wants to find an answer
            - ** current_slots **: record of filled slots in the dialogue so far
            
        ** return **: a dictionary of filled slots
        """
        logging.info('Calling `GOKBHelper` fill_inform_slots method')
        logging.debug("Inform slots to be  filled: '{0}'".format(self.pp.pformat(inform_slots_to_be_filled)))
        logging.debug("Current slots '{0}'".format(self.pp.pformat(current_slots)))

        # Get the available entities based on the history
        kb_results = self.available_results_from_kb(current_slots)
        filled_in_slots = {}

        # this happens in the end
        if const.TASK_COMPLETE_SLOT in inform_slots_to_be_filled.keys():
            filled_in_slots.update(current_slots[const.INFORM_SLOTS_KEY])

        # iterate over the inform slots
        for slot in inform_slots_to_be_filled.keys():

            # check in the special slots to check, if any
            if slot in self.special_slots:

                if slot in current_slots[const.INFORM_SLOTS_KEY].keys():
                    filled_in_slots[slot] = current_slots[const.INFORM_SLOTS_KEY][slot]
                # so stupid to check, of course it is in the list, we iterate over there
                elif slot in inform_slots_to_be_filled.keys():
                    filled_in_slots[slot] = inform_slots_to_be_filled[slot]

                continue

            # if the slot is the ultimate one or the one indicating the task is completed
            if slot == self.ultimate_request_slot or slot == const.TASK_COMPLETE_SLOT:
                filled_in_slots[slot] = const.TICKET_AVAILABLE if len(kb_results) > 0 else const.NO_VALUE_MATCH
                continue

            # how can I interpret this shit?
            if slot == 'closing': continue

            # Take the slot with the highest count and fill it
            values_dict = self.available_slot_values(slot, kb_results)
            values_counts = [(v, values_dict[v]) for v in values_dict.keys()]

            # if there are any results
            if len(values_counts) > 0:
                filled_in_slots[slot] = sorted(values_counts, key=lambda x: -x[1])[0][0]
            else:
                filled_in_slots[slot] = const.NO_VALUE_MATCH

        logging.debug("Filled in slots '{0}'".format(self.pp.pformat(filled_in_slots)))
        return filled_in_slots

    def available_slot_values(self, slot, kb_results):
        """
        Return the set of values available for the slot based on the current constraints
        
        # Arguments:
            
            - ** slot **: 
            - ** kb_results **: the results from the KB
             
        ** return **: 
        """
        logging.info('Calling `GOKBHelper` available_slot_values method')
        slot_values = {}

        # iterate over the kb results
        for entity_id in kb_results.keys():

            # count
            if slot in kb_results[entity_id].keys():
                slot_val = kb_results[entity_id][slot]

                if slot_val in slot_values.keys():
                    slot_values[slot_val] += 1
                else:
                    slot_values[slot_val] = 1

        return slot_values

    def available_results_from_kb(self, current_slots):
        """
        Return the available entities in the knowledge base, based on the current constraints.
        
        # Arguments
         
            - ** current_slots **: record of filled slots in the dialogue so far
            - ** filter_slots **: list of slots to filter out
         
        ** return **:
        """
        logging.info('Calling `GOKBHelper` available_results_from_kb method')

        # the resulting list of the available entities
        result = []

        # take only the constraints
        current_slots = current_slots[const.INFORM_SLOTS_KEY]
        constrain_keys = current_slots.keys()

        # filter out the slots
        constrain_keys = filter(lambda k: k not in self.filter_slots, constrain_keys)
        constrain_keys = [k for k in constrain_keys if current_slots[k] != const.I_DO_NOT_CARE]

        # for the given query index set, are there any cached results
        query_idx_keys = frozenset(current_slots.items())
        cached_kb_ret = self.cached_kb[query_idx_keys]

        cached_kb_length = len(cached_kb_ret) if cached_kb_ret != None else -1

        # if there are cached results, return then
        if cached_kb_length > 0:
            return dict(cached_kb_ret)

        elif cached_kb_length == -1:
            return dict([])

        # if the results is having length 0, continue

        # iterate ovet the knowledge base
        for id in self.knowledge_dict.keys():
            # get all keys for the given id
            kb_keys = self.knowledge_dict[id].keys()

            # some strange condition for finding a match
            if len(set(constrain_keys).union(set(kb_keys)) ^ (set(constrain_keys) ^ set(kb_keys))) == len(
                    constrain_keys):

                match = True
                for idx, k in enumerate(constrain_keys):
                    if str(current_slots[k]).lower() == str(self.knowledge_dict[id][k]).lower():
                        continue
                    else:
                        match = False

                # if there is a match, cache the results and add the match in the results
                if match:
                    self.cached_kb[query_idx_keys].append((id, self.knowledge_dict[id]))
                    result.append((id, self.knowledge_dict[id]))

        # if not a single match
        if len(result) == 0:
            self.cached_kb[query_idx_keys] = None

        # convert to dictionary
        result = dict(result)
        return result

    def available_results_from_kb_for_slots(self, inform_slots):
        """
        Return the count statistics for each constraint in inform_slots
        
        # Arguments:
        
            - ** inform_slots **: 
            
        ** return **:
        """
        logging.info('Calling `GOKBHelper` available_results_from_kb_for_slots method')

        kb_results = {key: 0 for key in inform_slots.keys()}
        kb_results[const.KB_MATCHING_ALL_CONSTRAINTS_KEY] = 0

        # load cached results
        query_idx_keys = frozenset(inform_slots.items())
        cached_kb_slot_ret = self.cached_kb_slot[query_idx_keys]

        # if there are already cached results, return them
        if len(cached_kb_slot_ret) > 0:
            logging.debug("Cached results found: '{0}'".format(cached_kb_slot_ret[0]))
            return cached_kb_slot_ret[0]

        # iterate in the knowledge dictionary
        for entity_id in self.knowledge_dict.keys():
            # iterate over the inform slots
            all_slots_match = 1
            for slot in inform_slots.keys():

                if slot == self.ultimate_request_slot or inform_slots[slot] == const.I_DO_NOT_CARE:
                    continue

                # if there is a match count
                if slot in self.knowledge_dict[entity_id].keys():

                    if inform_slots[slot].lower() == self.knowledge_dict[entity_id][slot].lower():
                        kb_results[slot] += 1
                    else:
                        all_slots_match = 0
                else:
                    all_slots_match = 0

            kb_results[const.KB_MATCHING_ALL_CONSTRAINTS_KEY] += all_slots_match

        self.cached_kb_slot[query_idx_keys].append(kb_results)

        return kb_results

    def database_results_for_agent(self, current_slots):
        """
        A dictionary of the number of results matching each current constraint.
        
        # Arguments:
        
            - ** current_slots **: record of filled slots in the dialogue so far
            
        ** return **: 
        """
        logging.info('Calling `GOKBHelper` database_results_for_agent method')

        database_results = self.available_results_from_kb_for_slots(current_slots[const.INFORM_SLOTS_KEY])
        logging.debug("Data Base Results: '{0}'".format(database_results))

        return database_results


    def suggest_slot_values(self, request_slots, current_slots):
        """
        Return the suggest slot values
        
        # Arguments:
        
            - ** request_slots **: 
            - ** current_slots **:
             
        ** return **: 
        """
        logging.info('Calling `GOKBHelper` suggest_slot_values method')

        avail_kb_results = self.available_results_from_kb(current_slots)
        return_suggest_slot_vals = {}
        for slot in request_slots.keys():
            avail_values_dict = self.available_slot_values(slot, avail_kb_results)
            values_counts = [(v, avail_values_dict[v]) for v in avail_values_dict.keys()]

            if len(values_counts) > 0:
                return_suggest_slot_vals[slot] = []
                sorted_dict = sorted(values_counts, key=lambda x: -x[1])
                for k in sorted_dict: return_suggest_slot_vals[slot].append(k[0])
            else:
                return_suggest_slot_vals[slot] = []

        return return_suggest_slot_vals

