import json, copy, os
import cPickle as pickle

from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentDQN
from deep_dialog.usersims import RuleSimulator


import random
from time import sleep
from pprint import pprint
import timeit

def split_user_goals(all_user_goals):
    """
    Helper method to split the user goals in two sets of goals, with and without request slots
    """

    user_goals_no_req_slots = []
    user_goals_with_req_slots = []

    for user_goal in all_user_goals:
        if len(user_goal["request_slots"].keys()) == 0:
            user_goals_no_req_slots.append(user_goal)
        else:
            user_goals_with_req_slots.append(user_goal)

    return user_goals_no_req_slots, user_goals_with_req_slots

def create_training_testing_records(training_user_goals_portions):
    """
    Helper method to create the training and testing structures for saving
    the performances.
    """
    training_records = {}
    testing_records = {}

    for num_goals in training_user_goals_portions:

        # training records
        training_records[num_goals] = {}

        training_records[num_goals]["success_rate"] = []
        training_records[num_goals]["ave_turns"] = []
        training_records[num_goals]["ave_reward"] = []

        # testing records
        testing_records[num_goals] = {}

        testing_records[num_goals]["success_rate"] = []
        testing_records[num_goals]["ave_turns"] = []
        testing_records[num_goals]["ave_reward"] = []

    return training_records, testing_records





def select_random_user_goals(user_goals_no_req_slots, user_goals_with_req_slots, cardinality_no_req, cardinality_req):
    """
    Helper method to randomly select user goals
    """

    random_user_goals = {}
    random_user_goals['all'] = []

    # select randomly user goals without request slots
    random_user_goals['all'].extend(copy.deepcopy(random.sample(user_goals_no_req_slots, cardinality_no_req)))
    # select randomly user goals with request slots
    random_user_goals['all'].extend(copy.deepcopy(random.sample(user_goals_with_req_slots, cardinality_req)))


    return random_user_goals


def warm_start_simulation(dialogue_manager, agent, warmup_user_goals, nb_warm_start_episodes_no_req,
                          nb_warm_start_episodes_req, agt_type):
    """
    Helper method for warm starting the simulation.
    
    """

    # split the warmup user goals in goals with and without request slots
    warmup_user_goals_no_req, warmup_user_goals_req  = split_user_goals(warmup_user_goals)

    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_episodes = 0

    # first we do warming up with the user goals without request slot
    for i in xrange(nb_warm_start_episodes_no_req):
        # select a random warmup user goal without request slots
        warmup_user_goal_no_req = random.choice(warmup_user_goals_no_req)
        dialogue_manager.initialize_warmup_episode(warmup_user_goal_no_req)
        episode_over = False

        while (not episode_over):
            episode_over, reward = dialogue_manager.next_turn()
            cumulative_reward += reward

            if episode_over:
                cumulative_turns += dialogue_manager.state_tracker.turn_count

                if reward > 0:
                    successes += 1

        warm_start_run_episodes += 1

        if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
            break


    # then we do warming up with the user goals with request slots
    for j in xrange(nb_warm_start_episodes_req):
        # select a random user goal with request slots
        warmup_user_goal_req = random.choice(warmup_user_goals_req)
        dialogue_manager.initialize_warmup_episode(warmup_user_goal_req)
        episode_over = False

        while (not episode_over):
            episode_over, reward = dialogue_manager.next_turn()
            cumulative_reward += reward

            if episode_over:
                cumulative_turns += dialogue_manager.state_tracker.turn_count

                if reward > 0:
                    successes += 1

        warm_start_run_episodes += 1

        if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
            break


    res['success_rate'] = float(successes) / warm_start_run_episodes
    res['ave_reward'] = float(cumulative_reward) / warm_start_run_episodes
    res['ave_turns'] = float(cumulative_turns) / warm_start_run_episodes

    print (agt_type + " Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (
        i + j + 2, res['success_rate'], res['ave_reward'], res['ave_turns']))

    # print ("Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))

    print (agt_type + " total length of the experience pool is %s" % (len(agent.experience_replay_pool)))
    return res



def simulation_epoch(num_simulation_episodes, dialogue_manager, agt_type):
    """
    Helper method to run the simulation epoch. In this case,
    the agent should have predict_mode = True and warm_start = 2,
    so it will write in the memory.
    
    """

    # track performances
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in xrange(num_simulation_episodes):
        dialogue_manager.initialize_episode()
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialogue_manager.next_turn()
            cumulative_reward += reward

            if episode_over:
                if reward > 0:
                    successes += 1
                    print (agt_type + " simulation episode %s: Success" % (episode))
                else:
                    print (agt_type + " simulation episode %s: Fail" % (episode))

                cumulative_turns += dialogue_manager.state_tracker.turn_count

    # calculate the results
    res['success_rate'] = float(successes) / num_simulation_episodes
    res['ave_reward'] = float(cumulative_reward) / num_simulation_episodes
    res['ave_turns'] = float(cumulative_turns) / num_simulation_episodes

    # print the results
    print ("simulation success rate %s, ave reward %s, ave turns %s" % (
        res['success_rate'], res['ave_reward'], res['ave_turns']))

    return res

def run_episodes(num_episodes, num_simulation_episodes, dialogue_manager, agent, agt_type):
    """
    Helper method to train the agent in `num_episodes` episodes
    """
    batch_size = 16

    # Best  Performances
    best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'episode': 0}
    best_model = {}
    best_model['model'] = copy.deepcopy(agent.dqn.model)

    # Running performances through the episodes
    performance_records = {}
    performance_records['success_rate'] = {}
    performance_records['ave_turns'] = {}
    performance_records['ave_reward'] = {}


    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    # iterate over the episodes
    for episode in xrange(num_episodes):
        # print ("Episode: %s" % (episode))

        episode_over = False
        dialogue_manager.initialize_episode()

        # run the episode without writing in the memory, which means perdict_mode = False
        while (not episode_over):
            episode_over, reward = dialogue_manager.next_turn()
            cumulative_reward += reward

            if episode_over:
                if reward > 0:
                    # print ("Successful Dialog!")
                    successes += 1
                # else:
                    # print ("Failed Dialog!")

                cumulative_turns += dialogue_manager.state_tracker.turn_count


        # simulation mode, set predict_mode to True
        agent.predict_mode = True
        # run one simulation epoch witn `num_simulation_episodes` episodes
        simulation_res = simulation_epoch(num_simulation_episodes, dialogue_manager, agt_type)

        # save the performances for the current episode
        performance_records['success_rate'][episode] = simulation_res['success_rate']
        performance_records['ave_turns'][episode] = simulation_res['ave_turns']
        performance_records['ave_reward'][episode] = simulation_res['ave_reward']

        # if the success rate is the best so far and more than the threshold
        if simulation_res['success_rate'] >= best_res['success_rate']:
            if simulation_res['success_rate'] >= 0.30:
                # empty the memory and do one aditional epoch to fill the memory with good examples
                agent.experience_replay_pool = []
                simulation_epoch(num_simulation_episodes, dialogue_manager, agt_type)

        # save the best performance
        if simulation_res['success_rate'] > best_res['success_rate']:
            best_model['model'] = copy.deepcopy(agent.dqn.model)
            best_res['success_rate'] = simulation_res['success_rate']
            best_res['ave_reward'] = simulation_res['ave_reward']
            best_res['ave_turns'] = simulation_res['ave_turns']
            best_res['episode'] = episode

        # train the agent
        agent.clone_dqn = copy.deepcopy(agent.dqn)
        agent.train(batch_size, 1)

        # return the predict mode to False
        agent.predict_mode = False

        # print the results for the simulation epoch
        # print ("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (
        #        performance_records['success_rate'][episode], performance_records['ave_reward'][episode],
        #        performance_records['ave_turns'][episode], best_res['success_rate']))


        # print the progress
        # print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
            # episode + 1, num_episodes, successes, episode + 1, float(cumulative_reward) / (episode + 1),
            # float(cumulative_turns) / (episode + 1)))


    # at the end of the training process report results
    # print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
        # successes, num_episodes, float(cumulative_reward) / num_episodes, float(cumulative_turns) / num_episodes))

    return performance_records, best_res, best_model['model']


def run_test_episodes(test_user_goals, num_test_repetition, dialogue_manager, agt_type):
    # print ("Start testing")
    test_performance = {}
    test_performance['success_rate'] = 0
    test_performance['ave_turns'] = 0
    test_performance['ave_reward'] = 0


    # Repeat tests
    for num_test in range(num_test_repetition):
        # print ("Test: %s" % (num_test))

        # Performances in the current test
        test_successes = 0
        test_cumulative_reward = 0
        test_cumulative_turns = 0

        # Iterate over the user goals
        for test_user_goal in test_user_goals:
            dialogue_manager.initialize_test_episode(test_user_goal)
            episode_over = False

            while (not episode_over):

                episode_over, reward = dialogue_manager.next_turn()
                test_cumulative_reward += reward

                if episode_over:
                    if reward > 0:
                        # print (agt_type + " Successful Test Dialog!")
                        test_successes += 1
                    else:
                        # print (agt_type + " Failed Test Dialog!")

                        test_cumulative_turns += dialogue_manager.state_tracker.turn_count

        test_performance['success_rate'] += float(test_successes)/len(test_user_goals)
        test_performance['ave_turns'] += float(test_cumulative_turns)/len(test_user_goals)
        test_performance['ave_reward'] += float(test_cumulative_reward)/len(test_user_goals)

    test_performance['success_rate'] /= num_test_repetition
    test_performance['ave_turns'] /= num_test_repetition
    test_performance['ave_reward'] /= num_test_repetition

    return test_performance


def save_performance_records(path, filename, records):

    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "wb"))
        # print 'saved model in %s' % (filepath, )
    except Exception, e:
        # print 'Error: Writing model fails: %s' % (filepath, )
        print e


################################################################################
# General Settings for both agents and user simulators
################################################################################

start = timeit.timeit()

# maximum number of turns per dialogue
max_turn = 40
# number of training episodes in each iteration
num_episodes = 100
# print nothing in the console
run_mode = 3
# semantic level
act_level = 0
# the number of warm start episodes
warm_start_episodes = 120
# number of episodes in one simulation epoch
simulation_epoch_size = 100
# the threshold rate
success_rate_threshold = 0.3
# the number of times to repeat the test
num_test_repetition = 10

################################################################################
# Training and Testing User Goals
################################################################################

# the path to the full set of user goals
train_user_goals_file_path = "./deep_dialog/data/user_goals.p"
# read all user goals
train_user_goals = pickle.load(open(train_user_goals_file_path, 'rb'))
# split the training user goals in sets with request and without request slots
train_user_goals_no_req_slots, train_user_goals_with_req_slots = split_user_goals(train_user_goals)


# the path to the full set of testing goals
test_user_goals_file_path = "./deep_dialog/data/user_test_goals.p"
# read all test user goals
test_user_goals = pickle.load(open(test_user_goals_file_path, 'rb'))

################################################################################
# Set of Dialogue Acts and Slot Types
################################################################################

act_set_file_path = "./deep_dialog/data/dia_acts.txt"
act_set = text_to_dict(act_set_file_path)

slot_set_file_path = "./deep_dialog/data/slot_set.txt"
slot_set = text_to_dict(slot_set_file_path)

################################################################################
#   Knowledge Base
################################################################################

# Knowledge Base Path
kb_path = "./deep_dialog/data/rest_kb.p"
# load the knowledge base
kb = pickle.load(open(kb_path, 'rb'))

################################################################################
#  Create the both Agents: pretrained and not pretrained
################################################################################

## Common Parameters of the agents

epsilon = 0  # the probability in the epsilon-greedy policy
experience_replay_pool_size = 1000  # the size of the memory
dqn_hidden_size = 80  # the number of units in the hidden layer
batch_size = 16  # the number of batches to replay from the memory
predict_mode = False
gamma = 0.9  #
warm_start = 1  # warm up the agents by following rule-based policy

## Pre-trained agent parameters

pretrained_agent_params = {}
pretrained_agent_params['max_turn'] = max_turn  # the maximal number of turns
pretrained_agent_params['agent_run_mode'] = run_mode  # print nothing
pretrained_agent_params['agent_act_level'] = act_level  # use semantic level

pretrained_agent_params['epsilon'] = epsilon
pretrained_agent_params['experience_replay_pool_size'] = experience_replay_pool_size
pretrained_agent_params['dqn_hidden_size'] = dqn_hidden_size
pretrained_agent_params['batch_size'] = batch_size
pretrained_agent_params['predict_mode'] = predict_mode
pretrained_agent_params['gamma'] = gamma
pretrained_agent_params['warm_start'] = warm_start
pretrained_agent_params['trained_model_path'] = None
pretrained_agent_params['pretrained_model_path'] = \
    "./deep_dialog/checkpoints/rl_agent/movie_booking/noe2e/agt_9_191_500_0.87800.p"

# the parameters for the testing agent using the pretrained model
pretrained_testing_agent_params = copy.deepcopy(pretrained_agent_params)
pretrained_testing_agent_params['pretrained_model_path'] = None


## Not a pre-trained agent parameters
agent_params = {}
agent_params['max_turn'] = max_turn  # the maximal number of turns
agent_params['agent_run_mode'] = run_mode  # print nothing
agent_params['agent_act_level'] = act_level  # use semantic level

agent_params['epsilon'] = epsilon
agent_params['experience_replay_pool_size'] = experience_replay_pool_size
agent_params['dqn_hidden_size'] = dqn_hidden_size
agent_params['batch_size'] = batch_size
agent_params['predict_mode'] = predict_mode
agent_params['gamma'] = gamma
agent_params['warm_start'] = warm_start
agent_params['pretrained_model_path'] = None
agent_params['trained_model_path'] = None

# the parameters for the testing agent trained from scratch
testing_agent_params = copy.deepcopy(agent_params)

################################################################################
#  Create the user simulators for both Agents: pretrained and not pretrained
################################################################################

## Common parameters for both simulators

slot_err_probability = 0.0  # the probability to make an error of the slot
slot_err_mode = 0
intent_err_prob = 0.0  # the probability to makr an error of the intent
learning_phase = 'all'  # no splitting of the training user goals
mock_dictionary = {}  # the mock dictionary used for corrupting the input

## User simulator parameters for the pre-trained agent

pretrain_usersim_params = {}
pretrain_usersim_params['max_turn'] = max_turn
pretrain_usersim_params['simulator_run_mode'] = run_mode
pretrain_usersim_params['simulator_act_level'] = act_level

pretrain_usersim_params['slot_err_probability'] = slot_err_probability
pretrain_usersim_params['slot_err_mode'] = slot_err_mode
pretrain_usersim_params['intent_err_probability'] = intent_err_prob
pretrain_usersim_params['learning_phase'] = learning_phase

## User simulator parameters for testing the pretrained model
pretrain_testing_usersim_params = copy.deepcopy(pretrain_usersim_params)

## User simulator parameters for the agent trained from scratch
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['simulator_run_mode'] = run_mode
usersim_params['simulator_act_level'] = act_level

usersim_params['slot_err_probability'] = slot_err_probability
usersim_params['slot_err_mode'] = slot_err_mode
usersim_params['intent_err_probability'] = intent_err_prob
usersim_params['learning_phase'] = learning_phase

## User simulator parameters for testing the model from scratch
testing_usersim_params = copy.deepcopy(usersim_params)

################################################################################
# The entire process of training and testing iterations with different train set size
################################################################################


# the number of training user goals in each training setting
training_user_goals_portions = [5, 10, 20, 30, 50, 120]

training_records_pretrain, testing_records_pretrain = create_training_testing_records(training_user_goals_portions)
training_records_scratch, testing_records_scratch = create_training_testing_records(training_user_goals_portions)

# learning curve results for the last
full_pretrain_performances = []
full_scratch_performances = []


# the cardinalities of the user goals without and with request slots
training_user_goals_no_req_cardinality = [2, 3, 6, 9, 15]
training_user_goals_req_cardinality = [3, 7, 14, 21, 35]

# number of training and testing iterations for each portion
num_iterations = 100

for i, num_goals in enumerate(training_user_goals_portions):
    # reset the iteration tracking

    # start the training and testing iterations
    # in each iteration select random portion of training user goals
    for iteration in range(num_iterations):
        # print ("Start iteration: {0}".format(iteration))

        ################################################################################
        # Training process
        ################################################################################



        # randomly select a set of user goals
        curr_user_goals = {}
        if num_goals != 120:
            curr_user_goals = select_random_user_goals(train_user_goals_no_req_slots, train_user_goals_with_req_slots,
                                                    training_user_goals_no_req_cardinality[i],
                                                    training_user_goals_req_cardinality[i])
        else:
            curr_user_goals["all"] = []
            curr_user_goals["all"].extend(copy.deepcopy(train_user_goals))

        # create pretrain user simulator
        pretrain_user_sim = RuleSimulator(mock_dictionary, act_set, slot_set, copy.deepcopy(curr_user_goals), pretrain_usersim_params)
        # create not a pre-trained user simulator
        user_sim = RuleSimulator(mock_dictionary, act_set, slot_set, copy.deepcopy(curr_user_goals), usersim_params)

        # create the pre-trained agent
        pretrained_agent = AgentDQN(kb, act_set, slot_set, pretrained_agent_params)
        # create the agent from scratch
        agent = AgentDQN(kb, act_set, slot_set, agent_params)

        # create dialogue manager for pre-trained agent
        pretrain_dialog_manager = DialogManager(pretrained_agent, pretrain_user_sim, act_set, slot_set, kb)

        # create dialogue manager for not pre-trained agent
        dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, kb)

        # the warmup success rate of the pre-trained model
        pretrain_warmup_succ_rate = 0
        # the warmup success rate of the scratch model
        warmup_succ_rate = 0


        # warm-start the pre-trained agent
        pretrain_warmup_res = warm_start_simulation(pretrain_dialog_manager, pretrained_agent, copy.deepcopy(curr_user_goals["all"]), 2, 8, "pretrain")


        # warm-start the agent from scartch
        warmup_res = warm_start_simulation(dialog_manager, agent, copy.deepcopy(curr_user_goals["all"]), 10, 20, "scratch")


        # after the warming up, we can continue with the actual training
        # turn off the warming up
        pretrained_agent.warm_start = 2
        agent.warm_start = 2

        # train the pre-trained model
        pretrain_performances, pretrain_best_res, best_pretrain_model = run_episodes(num_episodes, simulation_epoch_size,
                                                                                pretrain_dialog_manager, pretrained_agent, "pretrain")

        # train the model from scratch
        scratch_performances, scratch_best_res, best_scratch_model = run_episodes(num_episodes, simulation_epoch_size,
                                                                                dialog_manager, agent, "scratch")


        # if we train on the full data set
        if num_goals == 120:
            full_pretrain_performances.append(pretrain_performances['success_rate'])
            full_scratch_performances.append(scratch_performances['success_rate'])

            # save learning curves after each run
            save_performance_records("./deep_dialog/checkpoints/results/", "full_pretrain_performances.json",
                                    full_pretrain_performances)
            save_performance_records("./deep_dialog/checkpoints/results/", "full_scratch_performances.json",
                                    full_scratch_performances)


        # record the performances for the current training iteration for the pretrained model
        training_records_pretrain[num_goals]["success_rate"].append(pretrain_best_res["success_rate"])
        training_records_pretrain[num_goals]["ave_reward"].append(pretrain_best_res["ave_reward"])
        training_records_pretrain[num_goals]["ave_turns"].append(pretrain_best_res["ave_turns"])

        # record the performances for the current training iteration for the scratch model
        training_records_scratch[num_goals]["success_rate"].append(scratch_best_res["success_rate"])
        training_records_scratch[num_goals]["ave_reward"].append(scratch_best_res["ave_reward"])
        training_records_scratch[num_goals]["ave_turns"].append(scratch_best_res["ave_turns"])

        # save the training records after each run
        save_performance_records("./deep_dialog/checkpoints/results/", "training_records_pretrain.json",
                                 training_records_pretrain)
        save_performance_records("./deep_dialog/checkpoints/results/", "training_records_scratch.json",
                                 training_records_scratch)


        ################################################################################
        # Testing process
        ################################################################################

        # create testing user simulator for the pretrained model
        pretrain_testing_user_sim = RuleSimulator(mock_dictionary, act_set, slot_set, curr_user_goals,
                                                  pretrain_testing_usersim_params)
        # create not a pre-trained user simulator
        testing_user_sim = RuleSimulator(mock_dictionary, act_set, slot_set, curr_user_goals,
                                         testing_usersim_params)

        # create the pre-trained agent
        pretrained_testing_agent = AgentDQN(kb, act_set, slot_set, pretrained_testing_agent_params)

        # set up the DQN for the pretrained agent
        pretrained_testing_agent.dqn.model = copy.deepcopy(best_pretrain_model)
        pretrained_testing_agent.clone_dqn = copy.deepcopy(pretrained_testing_agent.dqn)
        pretrained_testing_agent.predict_mode = True
        pretrained_testing_agent.warm_start = 2
        pretrained_testing_agent.testing = True

        # create the agent from scratch
        testing_agent = AgentDQN(kb, act_set, slot_set, testing_agent_params)

        # set up the DQN for the pretrained agent
        testing_agent.dqn.model = copy.deepcopy(best_scratch_model)
        testing_agent.clone_dqn = copy.deepcopy(testing_agent.dqn)
        testing_agent.predict_mode = True
        testing_agent.warm_start = 2
        testing_agent.testing = True

        # create dialogue manager for pre-trained agent
        pretrain_testing_dialog_manager = DialogManager(pretrained_testing_agent, pretrain_testing_user_sim, act_set, slot_set, kb)

        # create dialogue manager for not pre-trained agent
        testing_dialog_manager = DialogManager(testing_agent, testing_user_sim, act_set, slot_set, kb)


        # perform testing for the pre-trained model
        pretrain_test_performace = run_test_episodes(copy.deepcopy(test_user_goals), num_test_repetition, pretrain_testing_dialog_manager, "pretrain")

        # perform testing for the model from scratch
        scratch_test_performace = run_test_episodes(copy.deepcopy(test_user_goals), num_test_repetition, testing_dialog_manager, "scratch")

        # record the performances for the current training iteration for the pretrained model
        testing_records_pretrain[num_goals]["success_rate"].append(pretrain_test_performace["success_rate"])
        testing_records_pretrain[num_goals]["ave_reward"].append(pretrain_test_performace["ave_reward"])
        testing_records_pretrain[num_goals]["ave_turns"].append(pretrain_test_performace["ave_turns"])

        # record the performances for the current training iteration for the scratch model
        testing_records_scratch[num_goals]["success_rate"].append(scratch_test_performace["success_rate"])
        testing_records_scratch[num_goals]["ave_reward"].append(scratch_test_performace["ave_reward"])
        testing_records_scratch[num_goals]["ave_turns"].append(scratch_test_performace["ave_turns"])

        # save the testing records after each run
        save_performance_records("./deep_dialog/checkpoints/results/", "testing_records_pretrain.json",
                                testing_records_pretrain)
        save_performance_records("./deep_dialog/checkpoints/results/", "testing_records_scratch.json",
                                testing_records_scratch)

end = timeit.timeit()
# print ("Total time: {0}".format(end - start))