"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file
"""

from core import constants as const

import core.dst.state_tracker as state_trackers
import core.user.users as users

from nlp.nlu.nlu import nlu
from nlp.nlg.nlg import nlg

from rl.core import Env
import logging


class GOEnv(Env):
    """
    The Environment with which the agent is interacting with. It extends the keras-rl class Env.
    Therefore, the following methods are implemented:
    
    - `step`
    - `reset`
    - `render`
    - `close`
    - `seed`
    - `configure`
    
    # Class members:
    
    From `rl.Env` class:
    
        - ** reward_shape **: the shape of the reward matrix
        - ** action_space **: the space of possible actions
        - ** observation_space **: the space of observations to ... (have to find the exact meaning)
    
    Own:
    
        - ** simulation_mode **: the mode of the simulation, semantic frame or natural language sentences
        - ** is_training **: flag indicating the training/testing mode
        - ** max_nb_turns **: the maximal number of allowed dialogue turns. Afterwards, the dialogue is considered failed
        - ** usr **: a simulated or real user making a conversation with the agent
        - ** state_tracker **: the state tracker used for tracking the state of the dialogue
        - ** nlu_unit **: the NLU unit for transforming the user utterance to a dialogue act
        - ** nlg_unit **: the NLG unit for transforming the agent's action to a natural language sentence
        - ** act_set **: the set of all dialogue acts
        - ** slot_set **: the set of all dialogue slots
        - ** feasible_actions **: list of templates described as dictionaries, corresponding to each action the agent might take
                            (dict to be specified)
        - ** reward_success **: the signaled reward for successful dialogue
        - ** reward_failure **: the signaled reward for failed dialogue
        - ** reward_neutral **: the signaled reward for ongoing dialogue
    """

    def __init__(self, act_set=None, slot_set=None, goal_set=None, init_inform_slots=None, ultimate_request_slot=None,
                 feasible_actions=None, kb_helper=None, params=None, *args, **kwargs):
        """
        Constructor for the Environment class.
        """
        logging.info('Calling `GOEnv` constructor')

        # call super class constructor
        super(GOEnv, self).__init__(*args, **kwargs)

        self.simulation_mode = params[const.SIMULATION_MODE_KEY]
        self.is_training = params[const.IS_TRAINING_KEY]

        self.user_type = params[const.USER_TYPE_KEY]
        self.state_tracker_type = params[const.STATE_TRACKER_TYPE_KEY]

        self.act_set = act_set
        self.slot_set = slot_set
        self.goal_set = goal_set
        self.init_inform_slots = init_inform_slots
        self.ultimate_request_slot = ultimate_request_slot
        self.feasible_actions = feasible_actions

        self.current_turn_nb = 0
        self.max_nb_turns = params[const.MAX_NB_TURNS]

        self.reward_success = params[const.SUCCESS_REWARD_KEY]
        self.reward_failure = params[const.FAILURE_REWARD_KEY]
        self.reward_neutral = params[const.PER_TURN_REWARD_KEY]

        self.kb_helper = kb_helper

        self.nlu_path = params[const.NLU_PATH_KEY]

        self.diaact_nl_pairs_path = params[const.DIAACT_NL_PAIRS_PATH_KEY]
        self.nlg_path = params[const.NLG_PATH_KEY]

        # create the user
        self.user = self.__create_user(params)

        # create the state tracker
        self.state_tracker = self.__create_state_tracker(params)

        # create the nlu unit
        self.nlu_unit = self.__create_nlu_unit()

        # create the nlg unit
        self.nlg_unit = self.__create_nlg_unit()

    def __create_user(self, params):
        """
        Private helper method for creating a user.
        
        # Arguments:
        
            - ** params **: 
            
        ** return **: the newly created user
        """
        logging.info('Calling `GOEnv` __create_user method')

        user = None

        user_type_str = params[const.USER_TYPE_KEY]

        if user_type_str == const.RULE_BASED_USER:
            user = users.GORuleBasedUser(simulation_mode=self.simulation_mode, goal_set=self.goal_set,
                                         max_nb_turns=self.max_nb_turns, slot_set=self.slot_set,
                                         act_set=self.act_set, init_inform_slots=self.init_inform_slots,
                                         ultimate_request_slot=self.ultimate_request_slot)
        elif user_type_str == const.MODEL_BASED_USER:
            user_path = params[const.MODEL_BASED_USER_PATH_KEY]
            user = users.GOModelBasedUser(self.simulation_mode, self.goal_set, self.slot_set,
                                          self.act_set, user_path)
        elif user_type_str == const.REAL_USER:
            user = users.GORealUser(self.goal_set)
        else:
            raise Exception()

        return user

    def __create_state_tracker(self, params):
        """
        Private helper method for creating a state tracker.
        
        # Arguments:
        
            - ** params **:
        
        ** return **: the newly created state tracker
        """
        logging.info('Calling `GOEnv` __create_state_tracker method')

        state_tracker = None
        dst_type_str = params[const.STATE_TRACKER_TYPE_KEY]
        if dst_type_str == const.RULE_BASED_STATE_TRACKER:
            state_tracker = state_trackers.GORuleBasedStateTracker(self.act_set, self.slot_set, self.max_nb_turns,
                                                                   self.kb_helper)
        elif dst_type_str == const.MODEL_BASED_STATE_TRACKER:
            dst_path = params[const.MODEL_BASED_STATE_TRACKER_PATH_KEY]
            state_tracker = state_trackers.GOModelBasedStateTracker(self.act_set, self.slot_set, self.max_nb_turns,
                                                                    self.kb_helper, dst_path)
        else:
            raise Exception()

        return state_tracker

    def __create_nlu_unit(self):
        """
        Private helper method for creating an NLU unit
        
        :return: the newly created NLU unit
        """
        logging.info('Calling `GOEnv` __create_nlu_unit method')

        nlu_unit = nlu()
        nlu_unit.load_nlu_model(self.nlu_path)

        return nlu_unit

    def __create_nlg_unit(self):
        """
        Private helper method for creating an NLU unit.
        
        :return: the newly created NLG unit
        """
        logging.info('Calling `GOEnv` __create_nlg_unit method')

        nlg_unit = nlg()
        nlg_unit.load_nlg_model(self.nlg_path)
        nlg_unit.load_predefine_act_nl_pairs(self.diaact_nl_pairs_path)

        return nlg_unit

    def __process_usr_action(self, usr_action):
        """
        Private helper method for processing the user action.
        
        :param usr_action: the user action to be processed
        :return: processed user action
        """
        logging.info('Calling `GOEnv`  __process_usr_action method')

        # by default add NL representation to the user action
        user_nlg_sentence = self.nlg_unit.convert_diaact_to_nl(usr_action, const.USR_SPEAKER_VAL)
        usr_action[const.NL_KEY] = user_nlg_sentence

        # if the simulation mode is on Natural Language level, generate new user action
        if self.simulation_mode == const.NL_SIMULATION_MODE:
            user_nlu_res = self.nlu_unit.generate_dia_act(usr_action[const.NL_KEY])
            usr_action.update(user_nlu_res)

        return usr_action

    def __process_agt_action(self, agt_action):
        """
        Private helper method for processing the agent action.
        
        :param agt_action: the agent action to be processed
        :return: processed agent action
        """
        logging.info('Calling `GOEnv`  __process_agt_action method')

        # add NL representation to the agent action
        agent_nlg_sentence = self.nlg_unit.convert_diaact_to_nl(agt_action, const.AGT_SPEAKER_VAL)
        agt_action[const.NL_KEY] = agent_nlg_sentence

        return agt_action

    def get_state_dimension(self):
        """
        
        ** return **: the dimension of the dialogue state
        """
        logging.info('Calling `GOEnv`  get_state_dimension method')

        return self.state_tracker.get_state_dimension()

    def get_current_turn_nb(self):
        """
        
        :return: 
        """
        return self.current_turn_nb

    def reward_function(self, dialogue_status):
        """
        
        # Arguments:
        
            - ** dialogue_status **: the status of the dialogue which can be one of the following:
            
                - ** SUCCESS_DIALOG **: successful dialogue, the user reached the goal
                - ** FAILED_DIALOG **: the dialogue failed, the user didn't reached the goal
                - ** NO_OUTCOME_YET **: the on-going dialogue
                
        ** return **: the reward associated with each status 
        """
        logging.info('Calling `GOEnv`  reward_function method')

        if dialogue_status == const.FAILED_DIALOG:
            return self.reward_failure
        elif dialogue_status == const.SUCCESS_DIALOG:
            return self.reward_success
        else:
            return self.reward_neutral

    def step(self, action):
        """
        Method for taking the environment one step further. In this case, to present the agent action to the user in order
        to make a response. Overrides the super class method.
        
        # Arguments:
        
            - ** action **: the last agent action
             
         ** return **: user's response to the agent's action in form of a state
        """
        logging.info('Calling `GOEnv` step method')

        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################

        self.current_turn_nb += 1
        # process the agent action
        proc_agt_action = self.__process_agt_action(action)
        # update the state tracker with the new agent action
        self.state_tracker.update(proc_agt_action, const.AGT_SPEAKER_VAL)


        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################

        # get the new user action and the dialogue status
        # The user signals if she reached the goal
        new_user_action, done, dialogue_status = self.user.step(proc_agt_action)
        reward = self.reward_function(dialogue_status)

        # if the user terminated the conversation
        if done:
            new_state = self.state_tracker.produce_state()
        else:
            # increase the dialogue turn number
            self.current_turn_nb += 1
            # process the new user action
            proc_new_user_action = self.__process_usr_action(new_user_action)
            # update the state tracker with the new user action
            self.state_tracker.update(proc_new_user_action, const.USR_SPEAKER_VAL)
            # produce new state for the
            new_state = self.state_tracker.produce_state()


        info = {}
        return new_state, reward, done, info

    def reset(self):

        """
        Method for resetting the dialogue state tracker and the user, called at the beginning of each new episode.
        Overrides the super class method.
        
        :return: the initial observation
        """

        logging.info('Calling `GOEnv` reset method')

        # reset the dst
        self.state_tracker.reset()
        # reset the user and get the initial action
        init_usr_action = self.user.reset()
        # initialize the number of turns
        self.current_turn_nb = 0
        # increase the dialogue turn number
        self.current_turn_nb += 1
        # process the init user action
        proc_init_usr_action = self.__process_usr_action(init_usr_action)
        # update the dialogue state tracker
        self.state_tracker.update(proc_init_usr_action, const.USR_SPEAKER_VAL)
        # produce state for the agent
        init_state = self.state_tracker.produce_state()

        return init_state

    def render(self, mode='human', close=False):
        # TODO
        raise NotImplementedError()

    def close(self):
        return True

    def seed(self, seed=None):
        # TODO
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        # TODO
        raise NotImplementedError()
