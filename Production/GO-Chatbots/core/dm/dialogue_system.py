"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the entire Goal-Oriented Dialogue System.
"""

from core import constants as const
from core.environment.environment import GOEnv
import core.agent.agents as agents
from core.agent.processor import GOProcessor
from core.dm.kb_helper import GOKBHelper
import cPickle as pickle
import logging
from keras.optimizers import Adam
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class GODialogSys(object):
    """
    The GO Dialogue System mediates the interaction between the environment and the agent.
    
    # Class members: 
    
        - ** agent **: the type of conversational agent. Default is None (temporarily).
        - ** environment **: the environment with which the agent and user interact. Default is None (temporarily).
        - ** act_set **: static set of all dialogue acts (intents) used in the dialogue. This set includes the following:
            
            - ** request **: the dialogue turn is requesting a value for some slots
            - ** inform **: the dialogue turn is providing values (constraints) for some values
            - ** confirm_question **:
            - ** confirm_answer **: 
            - ** greeting **: the turn does not provide any info else than a greeting
            - ** closing **: the turn
            - ** multiple_choice **: when the turn includes
            - ** thanks **: the turn does not provide any info else than a thanks words
            - ** welcome **: the turn does not provide any info else than a welcoming words
            - ** deny **:
            - ** not_sure **:
        - ** slot_set **: the set of all slots used in the dialogue.
        - ** knowledge_dict_path **: path to any knowledge dictionary for the database
        - ** agt_feasible_actions **: list of templates described as dictionaries, corresponding to each action the agent might take
                                (dict to be specified)
        - ** max_nb_turns **: the maximal number of dialogue turns
        - ** ultimate_request_slot **: the slot that is the actual goal of the user, and everything is around this slot.
    """

    def __init__(self, act_set=None, slot_set=None, goal_set=None, init_inform_slots=None, ultimate_request_slot=None,
                 kb_special_slots=None, kb_filter_slots=None, agt_feasible_actions=None, agt_memory=None,
                 agt_policy=None, agt_warmup_policy=None, agt_eval_policy=None, params=None):
        """
         Constructor of the class.
        """
        logging.info('Calling `GODialogSys` constructor')

        # Initialize the act set and slot set
        self.act_set = act_set
        self.slot_set = slot_set
        self.goal_set = goal_set
        self.init_inform_slots = init_inform_slots
        self.ultimate_request_slot = ultimate_request_slot
        self.kb_special_slots = kb_special_slots
        self.kb_filter_slots = kb_filter_slots

        # maximal number of turns
        self.max_nb_turns = params[const.MAX_NB_TURNS]

        # create the knowledge base helper class
        self.knowledge_dict = pickle.load(open(params[const.KB_PATH_KEY], 'rb'))
        self.kb_helper = GOKBHelper(self.ultimate_request_slot, self.kb_special_slots, self.kb_filter_slots,
                                    self.knowledge_dict)
        self.agt_feasible_actions = agt_feasible_actions

        # create the environment
        self.env = self.__create_env(params)

        # agent-related
        self.go_processor = GOProcessor(feasible_actions=self.agt_feasible_actions)
        self.nb_actions = len(self.agt_feasible_actions)
        self.agt_memory = agt_memory
        self.gamma = params[const.GAMMA_KEY]
        self.batch_size = params[const.BATCH_SIZE_KEY]
        self.nb_steps_warmup = params[const.NB_STEPS_WARMUP_KEY]
        self.train_interval = params[const.TRAIN_INTERVAL_KEY]
        self.memory_interval = params[const.MEMORY_INTERVAL_KEY]
        self.target_model_update = params[const.TARGET_MODEL_UPDATE_KEY]
        self.agt_policy = agt_policy
        self.agt_warmup_policy = agt_warmup_policy
        self.agt_eval_policy = agt_eval_policy
        self.enable_double_dqn = params[const.ENABLE_DOUBLE_DQN_KEY]
        self.enable_dueling_network = params[const.ENABLE_DUELING_NETWORK_KEY]
        self.dueling_type = params[const.DUELING_TYPE_KEY]
        self.state_dimension = self.env.get_state_dimension()
        self.hidden_size = params[const.HIDDEN_SIZE_KEY]
        self.act_func = params[const.ACTIVATION_FUNCTION_KEY]

        # create the specified agent type
        self.agent = self.__create_agent(params)

    def __create_env(self, params):
        """
        Private helper method for creating an environment given the parameters.
        
        # Arguments:
        
            - ** params **: the params for creating the environment
            
        ** return **: the newly created environment
        """
        logging.info('Calling `GODialogSys` __create_env method')

        # Create the environment
        env = GOEnv(self.act_set, self.slot_set, self.goal_set, self.init_inform_slots, self.ultimate_request_slot,
                    self.agt_feasible_actions, self.kb_helper, params)

        return env

    def __create_agent(self, params):
        """
        Private helper method for creating an agent depending on the given type as a string.
        
        :return: the newly created agent
        """
        logging.info('Calling `GODialogSys` __create_agent method')

        agent = None
        agent_type_value = params[const.AGENT_TYPE_KEY]

        if agent_type_value == const.AGENT_TYPE_DQN:
            agent = agents.GODQNAgent(processor=self.go_processor, nb_actions=self.nb_actions, memory=self.agt_memory,
                                      gamma=self.gamma, batch_size=self.batch_size,
                                      nb_steps_warmup=self.nb_steps_warmup,
                                      train_interval=self.train_interval, memory_interval=self.memory_interval,
                                      target_model_update=self.target_model_update, policy=self.agt_policy,
                                      warmup_policy=self.agt_warmup_policy,
                                      eval_policy=self.agt_eval_policy, enable_double_dqn=self.enable_double_dqn,
                                      enable_dueling_network=self.enable_dueling_network,
                                      dueling_type=self.dueling_type,
                                      output_dim=self.nb_actions, state_dimension=self.state_dimension,
                                      hidden_size=self.hidden_size, act_func=self.act_func)

            agent.compile(Adam(lr=.00025), metrics=['mae'])

        return agent

    def train(self, nb_epochs, nb_warmup_episodes, nb_episodes_per_epoch, res_path, weights_file_name):
        """
        Method for training the system.
        
        # Arguments:
        """

        self.agent.fit(env=self.env, nb_epochs=nb_epochs, nb_warmup_episodes=nb_warmup_episodes,
                       nb_episodes_per_epoch=nb_episodes_per_epoch, res_path=res_path)

        self.agent.save_weights(weights_file_name, overwrite=True)

    def initialize(self):
        """
        Method for initializing the dialogue
        """
