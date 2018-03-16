"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

"""

########################################################################################################################
# Agent-related constants                                                                                              #
########################################################################################################################

# key for specifying the type of the agent
AGENT_TYPE_KEY = "agent_type"
# value for the dqn agent type
AGENT_TYPE_DQN = "agent_type_dqn"

# Activation functions
RELU = 'relu'
SIGMOID = 'sigmoid'
TANH = 'tanh'
LINEAR = 'linear'

GAMMA_KEY = 'gamma'
BATCH_SIZE_KEY = 'batch_size'
NB_STEPS_WARMUP_KEY = 'nb_steps_warmup'
TRAIN_INTERVAL_KEY = 'train_interval'
MEMORY_INTERVAL_KEY = 'mamory_interval'
TARGET_MODEL_UPDATE_KEY = 'target_model_update'
HIDDEN_SIZE_KEY = 'hidden_size'
ACTIVATION_FUNCTION_KEY = 'activation_function'
ENABLE_DOUBLE_DQN_KEY = 'enable_double_dqn'
ENABLE_DUELING_NETWORK_KEY = 'enable_dueling_network'
DUELING_TYPE_KEY = 'dueling_type'



########################################################################################################################
# User-related constants                                                                                               #
########################################################################################################################

# key for specifying the type of the user
USER_TYPE_KEY = "user_type"
# value for the rule-based user type
RULE_BASED_USER = "rule_based_user"
# value for the model-based user type
MODEL_BASED_USER = "model_based_user"
# key for specifying a path to an already trained model-based user
MODEL_BASED_USER_PATH_KEY = "model_based_user_path"
# value for the real user type
REAL_USER = "real_user"
# key for specifying the user inform slots in the user internal state
USER_STATE_INFORM_SLOTS="user_inform_slots"
# key for specifying the user request slots in the user internal state
USER_STATE_REQUEST_SLOTS="user_request_slots"
# key for specifying the history of all user inform slots in the user internal state
USER_STATE_HISTORY_SLOTS="user_history_slots"
# key for specifying the history of all user slots in the user internal state
USER_STATE_REST_SLOTS="user_rest_slots"

########################################################################################################################
# State Tracker-related constants                                                                                      #
########################################################################################################################

# key for specifying the type of the state tracker
STATE_TRACKER_TYPE_KEY = "state_tracker_type"
# value for the rule-based state-tracker type
RULE_BASED_STATE_TRACKER = "rule_based_state_tracker"
# value for the model-based state-tracker type
MODEL_BASED_STATE_TRACKER = "model_based_state_tracker"
# key for specifying a path to an already trained model-based state tracker
MODEL_BASED_STATE_TRACKER_PATH_KEY = "model_based_state_tracker_path"

########################################################################################################################
# Agent training related constants                                                                                     #
########################################################################################################################

# key for specifying the simulation mode
SIMULATION_MODE_KEY = "simulation_mode"
# value for the semantic frame simulation mode
SEMANTIC_FRAME_SIMULATION_MODE = "semantic_frame_simulation_mode"
# value for the natural language simulation mode
NL_SIMULATION_MODE = "nl_simulation_mode"
# flag indicating the mode of the dialogue system
IS_TRAINING_KEY = "is_training"
# key for specifying the maximal number of dialogue turns
MAX_NB_TURNS = "max_nb_turns"


# key for specifying the path to the nlu unit
NLU_PATH_KEY = "nlu_path"

# key for specifying the path to the dialogue act paris
DIAACT_NL_PAIRS_PATH_KEY = "diaact_nl_pairs_path"
# key for specifying the path to the nlg unit
NLG_PATH_KEY = "nlg_path"


########################################################################################################################
# User and Agent action related constants                                                                              #
########################################################################################################################
# key for specifying the intent (act) of the dialogue turn
DIA_ACT_KEY = "diaact"
# key for specifying the inform slots
INFORM_SLOTS_KEY = "inform_slots"
# key for specifying the requested slots
REQUEST_SLOTS_KEY = "request_slots"
# key for specifying the task complete slot
TASK_COMPLETE_SLOT = "taskcomplete"
# key for specifying the nl part of the action
NL_KEY = "nl"
# key for specifying a proposed slot
PROPOSED_SLOT_KEY = "proposed_slots"
# key for specifying an agent requested slot
AGENT_REQUESTED_SLOT_KEY = "agent_request_slots"
# value for the unknown slots
UNKNOWN_SLOT_VALUE = "UKN"
# key for specifying the speaker type (user or agent)
SPEAKER_TYPE_KEY = "speaker"
# value for the speaker key, when the user is the speaker
USR_SPEAKER_VAL = "usr"
# value for the speaker key, when the agent is the speaker
AGT_SPEAKER_VAL = "agt"
# key for specifying the turn number
TURN_NB_KEY = "turn"

########################################################################################################################
# Knowledge Base related constants                                                                                     #
########################################################################################################################

# key for specifying a path to the knowledge base
KB_PATH_KEY = "kb_path"
# key for specifying a kb querying result where all of the constraints were matched
KB_MATCHING_ALL_CONSTRAINTS_KEY = "matching_all_constraints"

########################################################################################################################
# Dialog status related constants                                                                                      #
########################################################################################################################

# dialogue status
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = -1

SUCCESS_REWARD_KEY = "success_reward"
FAILURE_REWARD_KEY = "failure_reward"
PER_TURN_REWARD_KEY = "per_turn_reward"

########################################################################################################################
# all dialogue acts                                                                                                    #
########################################################################################################################

# key for specifying request dialogue act
REQUEST_DIA_ACT_KEY = "request"
# key for specifying inform dialogue act
INFORM_DIA_ACT_KEY = "inform"
# key for specifying confirm question dialogue act
CONFIRM_QUESTION_DIA_ACT_KEY = "confirm_question"
# key for specifying confirm answer dialogue act
CONFIRM_ANSWER_DIA_ACT_KEY = "confirm_answer"
# key for specifying greeting dialogue act
GREETING_DIA_ACT_KEY = "greeting"
# key for specifying closing dialogue act
CLOSING_DIA_ACT_KEY = "closing"
# key for specifying multiple choice dialogue act
MULTIPLE_CHOICE_DIA_ACT_KEY = "multiple_choice"
# key for specifying thanks dialogue act
THANKS_DIA_ACT_KEY = "thanks"
# key for specifying welcome dialogue act
WELCOME_DIA_ACT_KEY = "welcome"
# key for specifying deny dialogue act
DENY_DIA_ACT_KEY = "deny"
# key for specifying not sure dialogue act
NOT_SURE_DIA_ACT_KEY = "not_sure"


########################################################################################################################
#  Constraint Check                                                                                                    #
########################################################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

########################################################################################################################
#  Special Slot Values                                                                                                 #
########################################################################################################################
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
TICKET_AVAILABLE = 'Ticket Available'
