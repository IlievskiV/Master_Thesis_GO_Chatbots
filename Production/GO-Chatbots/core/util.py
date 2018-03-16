"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for utility methods in the Goal-Oriented Dialogue Systems
"""

import cPickle as pickle
import logging, os

project_path = os.path.join(os.path.dirname(__file__), '..')


def load_goal_set(goal_file_path):
    """ 
    Utility method to load the user goals
    
    # Arguments:
    
        - goal_file_path: the path to the user goals file
    """

    # Read all goals
    all_goal_set = pickle.load(open(goal_file_path, 'rb'))

    # Initialize and create the list of all goals
    goal_set = []
    for u_goal in all_goal_set:
        goal_set.append(u_goal)

    return goal_set


def text_to_dict(file_path):
    """
    Read in a text file as a dictionary where keys are text and values are indices (line numbers).
    Used to read the act set and slot set.
    
    # Arguments:
        
        - ** file_path **: The path to the act set or slot set file
        
    ** return **: string-index dictionary
    """


    result_set = {}
    with open(file_path, 'r') as f:
        index = 0
        for line in f.readlines():
            result_set[line.strip('\n').strip('\r')] = index
            index += 1

    return result_set