"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the users in the Goal-Oriented Dialogue Systems
"""

from core import constants as const
import random, copy, logging


class GOUser(object):
    """
    Abstract Base Class of all type of Goal-Oriented conversational users. The user is taking an action after each turn.
    One user action is represented as a dictionary having the exact same structure as the agent action, which is:
    
        - ** diaact **: the act of the action
        - ** inform_slots **: the set of informed slots
        - ** request_slots **: the set of request slots
        - ** nl **: the natural language representation of the agent action
    
    Also, the user is having a goal to follow, which is also a dictionary with the same structure as the user or agent
    action, with the difference that all information is provided.
    
    Moreover, the agent is keeping its own internal state, for its last turn, represented as a dictionary, which is:
    
        - ** diaact **: the user's dialogue act
        - ** inform_slots **: the user's inform slots from its previous turn
        - ** request_slots **: the user's request slots from its previous turn
        - ** history_slots **: the history of all user's inform slots
        - ** rest_slots **: the history of all user's slots
        
    # Class members:
    
        - ** id **: the id of the user
        - ** current_turn_nb **: the number of the current dialogue turn
        - ** state **: user internal state, keeping record of the past and current actions
        - ** simulation_mode **: semantic frame or natural language sentence form of user utterances
        - ** goal_set **: the set of goals for the user
        - ** goal **: the user goal in the current dialogue turn
    """

    def __init__(self, simulation_mode=None, goal_set=None, max_nb_turns=0):
        logging.info('Calling `GOUser` constructor')

        # Initialize the class members
        self.current_turn_nb = 0
        self.max_nb_turns = max_nb_turns

        self.state = {}
        self.state[const.DIA_ACT_KEY] = ""
        self.state[const.USER_STATE_INFORM_SLOTS] = {}
        self.state[const.USER_STATE_REQUEST_SLOTS] = {}
        self.state[const.USER_STATE_HISTORY_SLOTS] = {}
        self.state[const.USER_STATE_REST_SLOTS] = {}

        self.simulation_mode = simulation_mode
        self.goal_set = goal_set

        self.goal = None

    def __log_user_goal(self, usr_goal):
        """
        Abstract method for logging the user goal
        
        # Arguments:
        
            - ** usr_goal **: The current user goal 
        """

        raise NotImplementedError()

    def __log_user_action(self, usr_action):
        """
        Abstract method for logging the user action
        
        # Arguments:
        
            - ** usr_action **: the last user action
        """
        raise NotImplementedError()

    def reset(self):
        """
        Abstract method for restarting the user and getting the initial user action.
        
        :return: The initial user action
        """
        raise NotImplementedError()

    def step(self, agt_action):
        """
        Abstract method for getting the next user action given the last agent action.
        
        :param agt_action: last agent action
        :return: next user action and the dialogue status
        """
        raise NotImplementedError()


class GORealUser(GOUser):
    """
    Class connecting a real user, writing on the standard input. Extends the `GOUser` class.
    
    # Class members:
    """

    def __init__(self, goal_set=None, max_nb_turns=0):
        super(GORealUser, self).__init__(const.NL_SIMULATION_MODE, goal_set, max_nb_turns)

        logging.info('Calling `GORealUser` constructor')

    def __log_user_goal(self, usr_goal):
        """
        Overrides the abstract method from the super class
        """

        # TODO
        raise NotImplementedError()

    def __log_user_action(self, usr_action):
        """
        Overrides the abstract method from the super class
        """

        # TODO
        raise NotImplementedError()

    def reset(self):
        logging.info('Calling `GORealUser` reset method')

        # TODO
        raise NotImplementedError()

    def step(self, agt_action):
        logging.info('Calling `GORealUser` step method')

        # TODO
        raise NotImplementedError()


class GOSimulatedUser(GOUser):
    """
    Abstract Base Class for all simulated users in the Goal-Oriented Dialogue Systems.
    Extends the `GOUser` class.
    
    # Class members:
        
        - ** slot_set **: the set of all slots in the dialogue scenario
        - ** act_set **: the set of all acts (intents) in the dialogue scenario
        - ** dialog_status **: the status of the dialogue from the user perspective. The user is deciding whether the
                               dialogue is finished or not. The dialogue status could have the following value:
        
            - ** NO_OUTCOME_YET **: the dialogue is still ongoing
            - ** SUCCESS_DIALOG **: the dialogue was successful, i.e. the user achieved the goal
            - ** FAILED_DIALOG **: the dialogue failed, i.e. the user didn't succeed to achieve the goal
    """

    def __init__(self, simulation_mode=None, goal_set=None, max_nb_turns=0, slot_set=None, act_set=None):
        super(GOSimulatedUser, self).__init__(simulation_mode, goal_set, max_nb_turns)
        logging.info('Calling `GOSimulatedUser` constructor')

        self.slot_set = slot_set
        self.act_set = act_set

        self.dialog_status = const.NO_OUTCOME_YET

    def __sample_random_init_action(self):
        """
        Abstract private helper method for sampling a random initial user action based on the goal.

        :return: random initial user action
        """
        raise NotImplementedError()

    def __sample_goal(self):
        """
        Abstract private helper method for sampling a random user goal, given the set of all available user goals.

        :return: random user goal
        """
        raise NotImplementedError()

    def __log_user_goal(self, usr_goal):
        """
        Overrides the abstract method from the super class
        """

        raise NotImplementedError()

    def __log_user_action(self, usr_action):
        """
        Overrides the abstract method from the superclass
        """
        raise NotImplementedError()

    def reset(self):
        """
        Overrides the abstract method from the super class
        """

        logging.info('Calling `GOSimulatedUser` reset method')
        raise NotImplementedError()

    def step(self, agt_action):
        """
        Abstract method from the super class
        """

        logging.info('Calling `GOSimulatedUser` step method')
        raise NotImplementedError()


class GORuleBasedUser(GOSimulatedUser):
    """
    Abstract class representing a rule-based user in the Goal-Oriented Dialogue Systems.
    Since, it is a rule-based simulated user, it will be domain-specific.
    Extends the `GOUser` class.
    
    Class members:
    
        - ** init_inform_slots **: list of initial inform slots, such that if the current user goal is containing some
                                   of them, they must appear in the initial user turn
                                   
        - ** ultimate_request_slot ** : the slot that is the actual goal of the user, and everything is around this slot.
    """

    def __init__(self, simulation_mode=None, goal_set=None, max_nb_turns=0, slot_set=None, act_set=None,
                 init_inform_slots=None, ultimate_request_slot=None):
        super(GORuleBasedUser, self).__init__(simulation_mode, goal_set, max_nb_turns, slot_set, act_set)

        logging.info('Calling `GORuleBasedUser` constructor')

        self.init_inform_slots = init_inform_slots
        self.ultimate_request_slot = ultimate_request_slot

    def __log_user_goal(self, usr_goal):
        """
        Overrides the abstract method from the super class
        """

        logging.info("The `GORuleBaseUser` class user goal: ")
        logging.info("Inform slots: ")

        for slot in usr_goal[const.INFORM_SLOTS_KEY].keys():
            logging.info("\t\t '{0}'='{1}'".format(slot, usr_goal[const.INFORM_SLOTS_KEY][slot]))

        logging.info("Request slots: ")

        for slot in usr_goal[const.REQUEST_SLOTS_KEY].keys():
            logging.info("\t\t '{0}'='{1}'".format(slot, usr_goal[const.REQUEST_SLOTS_KEY][slot]))

    def __log_user_action(self, usr_action):
        """
        Overrides the abstract method from the superclass
        """

        logging.info("The `GORuleBaseUser` class user action: ")
        logging.info("\t Dialogue Act: '{0}'".format(usr_action[const.DIA_ACT_KEY]))
        logging.info("Inform slots: ")

        for slot in usr_action[const.INFORM_SLOTS_KEY].keys():
            logging.info("\t\t '{0}'='{1}'".format(slot, usr_action[const.INFORM_SLOTS_KEY][slot]))

        logging.info("Request slots: ")

        for slot in usr_action[const.REQUEST_SLOTS_KEY].keys():
            logging.info("\t\t '{0}'='{1}'".format(slot, usr_action[const.REQUEST_SLOTS_KEY][slot]))

    def __sample_random_init_action(self):
        """
        Overrides abstract method from the super class
        """

        logging.info('Calling `GORuleBasedUser` __sample_random_init_action method')

        # increase the dialogue number turn
        self.current_turn_nb += 1

        # the resulting initial action
        init_action = {}

        # the initial dialogue act is request. If no request slots, it will become inform
        self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY

        # sample inform slots
        if len(self.goal[const.INFORM_SLOTS_KEY]) > 0:
            # sample an inform slot from the current user goal and insert it in the user's internal state
            sampled_inform_slot = random.choice(list(self.goal[const.INFORM_SLOTS_KEY].keys()))
            self.state[const.USER_STATE_INFORM_SLOTS][sampled_inform_slot] = self.goal[const.INFORM_SLOTS_KEY][
                sampled_inform_slot]

            # after sampling the initial inform slot, check the presence of the initial slots
            # if a slot in the set of initial slots for the inform act and is in the current goal, it must appear
            for init_slot in self.init_inform_slots:
                if init_slot != sampled_inform_slot and init_slot in self.goal[const.INFORM_SLOTS_KEY].keys():
                    self.state[const.USER_STATE_INFORM_SLOTS][init_slot] = self.goal[const.INFORM_SLOTS_KEY][init_slot]

            # the inform slots in the goal, which are not in the list of init goals, put them in a list of rest slots
            for slot in self.goal[const.INFORM_SLOTS_KEY].keys():
                if sampled_inform_slot == slot or slot in self.init_inform_slots:
                    continue
                else:
                    self.state[const.USER_STATE_REST_SLOTS].append(slot)

        # extend the list of rest slots with the list of request slots in the user goal
        self.state[const.USER_STATE_REST_SLOTS].extend(self.goal[const.REQUEST_SLOTS_KEY].keys())

        # sample request slots

        # if there are some other request slots, we don't present the ultimate goal to the agent
        # however, if there is no other option, present the ultimate slot


        request_slot_set = list(self.goal[const.REQUEST_SLOTS_KEY].keys())
        if (self.ultimate_request_slot in request_slot_set):
            request_slot_set.remove(self.ultimate_request_slot)

        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = copy.deepcopy(self.ultimate_request_slot)

        self.state[const.USER_STATE_REQUEST_SLOTS][request_slot] = 'UNK'

        # if there are no request slots, we have inform dialogue act
        if len(self.state[const.USER_STATE_REQUEST_SLOTS]) == 0:
            self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY

        # create the user action
        init_action[const.DIA_ACT_KEY] = self.state[const.DIA_ACT_KEY]
        init_action[const.INFORM_SLOTS_KEY] = self.state[const.USER_STATE_INFORM_SLOTS]
        init_action[const.REQUEST_SLOTS_KEY] = self.state[const.USER_STATE_REQUEST_SLOTS]

        # log the init action
        self.__log_user_action(init_action)

        return init_action

    def __sample_goal(self):
        """
        Overrides the abstract method from the super class
        """

        logging.info('Calling `GORuleBasedUser` __sample_goal method')
        sample_goal = random.choice(self.goal_set)

        # log the user goal
        self.__log_user_goal(sample_goal)

        return sample_goal

    def __response_inform(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'inform'.
        
        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_inform method')
        # if the inform slots in the agent action contain 'task complete' slot, it means the agent completed the user task
        if const.TASK_COMPLETE_SLOT in agt_action[const.INFORM_SLOTS_KEY].keys():
            return self.__response_inform_task_complete(agt_action)

        # if the task is not completed
        else:
            self.__response_inform_task_not_complete(agt_action)

    def __response_inform_task_complete(self, agt_action):
        """
        Private helper method, called when the task is complete when the user is responding to an inform agent 
        dialogue act.
        
        :param agt_action: the last agent action
        :return: 
        """

        logging.info('Calling `GORuleBasedUser` __response_inform_task_complete method')
        # the next user action will be thanks
        self.state[const.DIA_ACT_KEY] = const.THANKS_DIA_ACT_KEY
        # the user has satisfied the required constraint, whether the required value was found or not
        self.constraint_check = const.CONSTRAINT_CHECK_SUCCESS

        # if the task was completed without match
        if agt_action[const.INFORM_SLOTS_KEY][const.TASK_COMPLETE_SLOT] == const.NO_VALUE_MATCH:
            self.state[const.USER_STATE_HISTORY_SLOTS][self.ultimate_request_slot] = const.NO_VALUE_MATCH

            # if the ultimate goal was in the rest slots or in the request slots, delete it
            if self.ultimate_request_slot in self.state[const.USER_STATE_REST_SLOTS]:
                self.state[const.USER_STATE_REST_SLOTS].remove(self.ultimate_request_slot)

            if self.ultimate_request_slot in self.state[const.USER_STATE_REQUEST_SLOTS].keys():
                del self.state[const.USER_STATE_REQUEST_SLOTS][self.ultimate_request_slot]

        for slot in self.goal[const.INFORM_SLOTS_KEY].keys():
            #  Deny, if the answers from agent can not meet the constraints of user
            if slot not in agt_action[const.INFORM_SLOTS_KEY].keys() or (
                        self.goal[const.INFORM_SLOTS_KEY][slot].lower() != agt_action[const.INFORM_SLOTS_KEY][
                        slot].lower()):
                self.state[const.DIA_ACT_KEY] = const.DENY_DIA_ACT_KEY
                self.state[const.USER_STATE_REQUEST_SLOTS].clear()
                self.state[const.USER_STATE_INFORM_SLOTS].clear()
                self.constraint_check = const.CONSTRAINT_CHECK_FAILURE
                break

        return True

    def __response_inform_task_not_complete(self, agt_action):
        """
        Private helper method, called when the task is NOT complete when the user is responding to an inform agent 
        dialogue act.
        
        :param agt_action: the last agent action
        :return: 
        """

        logging.info('Calling `GORuleBasedUser` __response_inform_task_not_complete method')
        # iterate over the agent inform slots
        for slot in agt_action[const.INFORM_SLOTS_KEY].keys():
            # put it in the history slot
            self.state[const.USER_STATE_HISTORY_SLOTS][slot] = agt_action[const.INFORM_SLOTS_KEY][slot]

            # now we should work on the next user action

            # if the agent inform slot is in the user goal inform slots
            if slot in self.goal[const.INFORM_SLOTS_KEY].keys():

                # if the agent inform slot value is equal to the value of the same slot in the user goal
                if agt_action[const.INFORM_SLOTS_KEY][slot] == self.goal[const.INFORM_SLOTS_KEY][slot]:

                    # if the slot is in the user rest slots, remove it
                    if slot in self.state[const.USER_STATE_REST_SLOTS]:
                        self.state[const.USER_STATE_REST_SLOTS].remove(slot)

                    # if the user has request slots in its state, the dialogue act is request
                    if len(self.state[const.USER_STATE_REQUEST_SLOTS]) > 0:
                        self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY

                    # it the user is having rest slots in its state
                    elif len(self.state[const.USER_STATE_REST_SLOTS]) > 0:

                        # if the ultimate slot is in the rest slots, remove it from a copy
                        rest_slot_set = copy.deepcopy(self.state[const.USER_STATE_REST_SLOTS])
                        if self.ultimate_request_slot in rest_slot_set:
                            rest_slot_set.remove(self.ultimate_request_slot)

                        # if the copy of the rest slots contain other slots than the ultimate
                        if len(rest_slot_set) > 0:
                            # randomly choose one =
                            inform_slot = random.choice(rest_slot_set)

                            # if the randomly drawn slot is in the inform slots of the user goal
                            if inform_slot in self.goal[const.INFORM_SLOTS_KEY].keys():
                                # put it in the user state inform slots and make a inform dialogue act
                                self.state[const.USER_STATE_INFORM_SLOTS][inform_slot] = \
                                    self.goal[const.INFORM_SLOTS_KEY][inform_slot]
                                self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY
                                # remove it from the use rstate rest slots
                                self.state[const.USER_STATE_REST_SLOTS].remove(inform_slot)

                            # if the randomly drawn slot is in the request slots of the user goal
                            elif inform_slot in self.goal[const.REQUEST_SLOTS_KEY].keys():
                                # put it in the user state request slots and make a request dialogue act
                                self.state[const.USER_STATE_REQUEST_SLOTS][inform_slot] = 'UNK'
                                self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY

                        # if in the rest slots there are no other slots than the ultimate
                        else:
                            # make the ultimate goal a request slot and make a request dialogue act
                            self.state[const.USER_STATE_REQUEST_SLOTS][self.ultimate_request_slot] = 'UNK'
                            self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY

                    # if there are no request and rest slots, the user says thanks
                    else:
                        self.state[const.DIA_ACT_KEY] = const.THANKS_DIA_ACT_KEY

                # if the agent inform slot value is NOT equal to the value of the same slot in the user goal
                else:
                    # make an inform dialogue act key with a correct information
                    self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY
                    self.state[const.USER_STATE_INFORM_SLOTS][slot] = self.goal[const.INFORM_SLOTS_KEY][slot]

                    if slot in self.state[const.USER_STATE_REST_SLOTS]:
                        self.state[const.USER_STATE_REST_SLOTS].remove(slot)

            # if the current agent slot is not in the set of user goal inform slots
            else:
                # if the slot is in the user state rest slots, remove it from there
                if slot in self.state[const.USER_STATE_REST_SLOTS]:
                    self.state[const.USER_STATE_REST_SLOTS].remove(slot)

                # if the slot is in the user state request slots, remove it from there too
                if slot in self.state[const.USER_STATE_REQUEST_SLOTS].keys():
                    del self.state[const.USER_STATE_REQUEST_SLOTS][slot]

                # chose from the request slots
                if len(self.state[const.USER_STATE_REQUEST_SLOTS]) > 0:

                    # if there are some other request slots, we don't present the ultimate goal to the agent
                    # however, if there is no other option, present the ultimate slot
                    request_set = list(self.state[const.USER_STATE_REQUEST_SLOTS].keys())
                    if self.ultimate_request_slot in request_set:
                        request_set.remove(self.ultimate_request_slot)

                    if len(request_set) > 0:
                        request_slot = random.choice(request_set)
                    else:
                        request_slot = copy.deepcopy(self.ultimate_request_slot)

                    # make a request dialogue act
                    self.state[const.USER_STATE_REQUEST_SLOTS][request_slot] = "UNK"
                    self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY

                # if there are no request slots, chose from the rest slots
                elif len(self.state[const.USER_STATE_REST_SLOTS]) > 0:

                    # if there are some other rest slots, we don't present the ultimate goal to the agent
                    # however, if there is no other option, present the ultimate slot
                    rest_slot_set = copy.deepcopy(self.state[const.USER_STATE_REST_SLOTS])
                    if self.ultimate_request_slot in rest_slot_set:
                        rest_slot_set.remove(self.ultimate_request_slot)

                    if len(rest_slot_set) > 0:
                        inform_slot = random.choice(rest_slot_set)

                        if inform_slot in self.goal[const.INFORM_SLOTS_KEY].keys():
                            self.state[const.USER_STATE_INFORM_SLOTS][inform_slot] = \
                                self.goal[const.INFORM_SLOTS_KEY][inform_slot]
                            self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY

                            self.state[const.USER_STATE_REST_SLOTS].remove(inform_slot)

                            if self.ultimate_request_slot in self.state[const.USER_STATE_REST_SLOTS]:
                                self.state[const.USER_STATE_REQUEST_SLOTS][self.ultimate_request_slot] = 'UNK'
                                self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY

                        elif inform_slot in self.goal[const.REQUEST_SLOTS_KEY].keys():
                            self.state[const.USER_STATE_REQUEST_SLOTS][inform_slot] = \
                                self.goal[const.REQUEST_SLOTS_KEY][inform_slot]
                            self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY

                    else:
                        self.state[const.USER_STATE_REQUEST_SLOTS][self.ultimate_request_slot] = 'UNK'
                        self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY

                # if nothing else say thanks
                else:
                    self.state[const.DIA_ACT_KEY] = const.THANKS_DIA_ACT_KEY

        return True

    def __response_request(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'request'.
        
        :param agt_action: the last agent action
        :return:
        """
        logging.info('Calling `GORuleBasedUser` __response_request method')
        # if the agent action contains request slots
        if len(agt_action[const.REQUEST_SLOTS_KEY].keys()) > 0:

            # take the first request slot
            slot = agt_action[const.REQUEST_SLOTS_KEY].keys()[0]

            # request slot in user's goal constraints
            if slot in self.goal[const.INFORM_SLOTS_KEY].keys():

                self.state[const.USER_STATE_INFORM_SLOTS][slot] = self.goal[const.INFORM_SLOTS_KEY][slot]
                self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY

                # remove it from the user state rest slots
                if slot in self.state[const.USER_STATE_REST_SLOTS]:
                    self.state[const.USER_STATE_REST_SLOTS].remove(slot)

                # remove ir from the user state request slots
                if slot in self.state[const.USER_STATE_REQUEST_SLOTS].keys():
                    del self.state[const.USER_STATE_REQUEST_SLOTS][slot]

                self.state[const.USER_STATE_REQUEST_SLOTS].clear()

            # the requested slot has been answered
            elif slot in self.goal[const.REQUEST_SLOTS_KEY].keys() and slot not in self.state[
                const.USER_STATE_REST_SLOTS] and slot in \
                    self.state[const.USER_STATE_HISTORY_SLOTS].keys():

                self.state[const.USER_STATE_INFORM_SLOTS][slot] = self.state[const.USER_STATE_HISTORY_SLOTS][slot]
                self.state[const.USER_STATE_REQUEST_SLOTS].clear()
                self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY

            # request slot in user's goal's request slots, and not answered yet
            elif slot in self.goal[const.REQUEST_SLOTS_KEY].keys() and slot in self.state[const.USER_STATE_REST_SLOTS]:

                self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY  # "confirm_question"
                self.state[const.USER_STATE_REQUEST_SLOTS][slot] = "UNK"

                for info_slot in self.state[const.USER_STATE_REST_SLOTS]:
                    if info_slot in self.goal[const.INFORM_SLOTS_KEY].keys():
                        self.state[const.USER_STATE_INFORM_SLOTS][info_slot] = self.goal[const.INFORM_SLOTS_KEY][
                            info_slot]

                for info_slot in self.state[const.USER_STATE_INFORM_SLOTS].keys():
                    if info_slot in self.state[const.USER_STATE_REST_SLOTS]:
                        self.state[const.USER_STATE_REST_SLOTS].remove(info_slot)

            else:

                if len(self.state[const.USER_STATE_REQUEST_SLOTS]) == 0 and len(
                        self.state[const.USER_STATE_REST_SLOTS]) == 0:
                    self.state[const.DIA_ACT_KEY] = const.THANKS_DIA_ACT_KEY
                else:
                    self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY

                self.state[const.USER_STATE_INFORM_SLOTS][slot] = const.I_DO_NOT_CARE

        else:
            if len(self.state[const.USER_STATE_REST_SLOTS]) > 0:
                random_slot = random.choice(self.state[const.USER_STATE_REST_SLOTS])

                if random_slot in self.goal[const.INFORM_SLOTS_KEY].keys():
                    self.state[const.USER_STATE_INFORM_SLOTS][random_slot] = self.goal[const.INFORM_SLOTS_KEY][
                        random_slot]

                    self.state[const.USER_STATE_REST_SLOTS].remove(random_slot)
                    self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY

                elif random_slot in self.goal[const.REQUEST_SLOTS_KEY].keys():
                    self.state[const.USER_STATE_REQUEST_SLOTS][random_slot] = self.goal[const.REQUEST_SLOTS_KEY][
                        random_slot]

                    self.state[const.DIA_ACT_KEY] = const.REQUEST_SLOTS_KEY

        return True

    def __response_confirm_question(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'confirm_question'.

        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_confirm_question method')
        # TODO
        return True

    def __response_confirm_answer(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'confirm_answer'.
        
        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_confirm_answer method')
        if len(self.state[const.USER_STATE_REST_SLOTS]) > 0:
            request_slot = random.choice(self.state[const.USER_STATE_REST_SLOTS])

            if request_slot in self.goal[const.REQUEST_SLOTS_KEY].keys():
                self.state[const.DIA_ACT_KEY] = const.REQUEST_DIA_ACT_KEY
                self.state[const.USER_STATE_REQUEST_SLOTS][request_slot] = "UNK"

            elif request_slot in self.goal[const.INFORM_SLOTS_KEY].keys():
                self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY
                self.state[const.USER_STATE_INFORM_SLOTS][request_slot] = self.goal[const.INFORM_SLOTS_KEY][
                    request_slot]

                if request_slot in self.state[const.USER_STATE_REST_SLOTS]:
                    self.state[const.USER_STATE_REST_SLOTS].remove(request_slot)

        # otherwise the next user action is thanks
        else:
            self.state[const.DIA_ACT_KEY] = const.THANKS_DIA_ACT_KEY

        return True

    def __response_greeting(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'greeting'. 

        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_greeting method')
        # TODO
        return True

    def __response_closing(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'closing'. 

        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_closing method')
        self.state[const.DIA_ACT_KEY] = const.THANKS_DIA_ACT_KEY
        return True

    def __response_multiple_choice(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'multiple_choice'.
        
        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_multiple_choice method')
        # take the first inform slot from the agent response
        slot = agt_action[const.INFORM_SLOTS_KEY].keys()[0]

        # the slot is in the user goal's inform slots, make it inform slot
        if slot in self.goal[const.INFORM_SLOTS_KEY].keys():
            self.state[const.USER_STATE_INFORM_SLOTS][slot] = self.goal[const.INFORM_SLOTS_KEY][slot]

        # the slot is in the user goal's request slots
        elif slot in self.goal[const.REQUEST_SLOTS_KEY].keys():
            self.state[const.USER_STATE_INFORM_SLOTS][slot] = random.choice(agt_action[const.INFORM_SLOTS_KEY][slot])

        # make an inform dialogue act
        self.state[const.DIA_ACT_KEY] = const.INFORM_DIA_ACT_KEY

        # delete it from the rest slots if any
        if slot in self.state[const.USER_STATE_REST_SLOTS]:
            self.state[const.USER_STATE_REST_SLOTS].remove(slot)

        # delete it from the request slots if any
        if slot in self.state[const.USER_STATE_REQUEST_SLOTS].keys():
            del self.state[const.USER_STATE_REQUEST_SLOTS][slot]

        return True

    def __response_thanks(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'thanks'.
        
        
        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_thanks method')
        self.dialog_status = const.SUCCESS_DIALOG

        # remove the ultimate slot from the user request slots
        request_slot_set = copy.deepcopy(self.state[const.USER_STATE_REQUEST_SLOTS].keys())
        if self.ultimate_request_slot in request_slot_set:
            request_slot_set.remove(self.ultimate_request_slot)

        # remove the ultimate slot from the user rest slots
        rest_slot_set = copy.deepcopy(self.state[const.USER_STATE_REST_SLOTS])
        if self.ultimate_request_slot in rest_slot_set:
            rest_slot_set.remove(self.ultimate_request_slot)

        # if the user had not answered
        if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
            self.dialog_status = const.FAILED_DIALOG

        # check the user history slots
        for info_slot in self.state[const.USER_STATE_HISTORY_SLOTS].keys():
            if self.state[const.USER_STATE_HISTORY_SLOTS][info_slot] == const.NO_VALUE_MATCH:
                self.dialog_status = const.FAILED_DIALOG

            if info_slot in self.goal[const.INFORM_SLOTS_KEY].keys():
                if self.state[const.USER_STATE_HISTORY_SLOTS][info_slot] != self.goal[const.INFORM_SLOTS_KEY][
                    info_slot]:
                    self.dialog_status = const.FAILED_DIALOG

        if self.ultimate_request_slot in agt_action[const.INFORM_SLOTS_KEY].keys():
            if agt_action[const.INFORM_SLOTS_KEY][self.ultimate_request_slot] == const.NO_VALUE_MATCH:
                self.dialog_status = const.FAILED_DIALOG

        if self.constraint_check == const.CONSTRAINT_CHECK_FAILURE:
            self.dialog_status = const.FAILED_DIALOG

        return True

    def __response_welcome(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'welcome'.


        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_welcome method')
        # TODO
        return True

    def __response_deny(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'deny'.


        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_deny method')
        # TODO
        return True

    def __response_not_sure(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'not_sure'.


        :param agt_action: the last agent action
        :return:
        """

        logging.info('Calling `GORuleBasedUser` __response_not_sure method')
        # TODO
        return True

    def reset(self):
        """
        Overrides the abstract method from the super class
        """

        logging.info('Calling `GORuleBasedUser` reset method')

        # reset the number of turns
        self.current_turn_nb = 0

        # reset the user state
        self.state = {}
        self.state[const.DIA_ACT_KEY] = ""
        self.state[const.USER_STATE_INFORM_SLOTS] = {}
        self.state[const.USER_STATE_REQUEST_SLOTS] = {}
        self.state[const.USER_STATE_HISTORY_SLOTS] = {}
        self.state[const.USER_STATE_REST_SLOTS] = []

        # set the dialogue status to ongoing
        self.episode_over = False
        self.dialog_status = const.NO_OUTCOME_YET
        self.constraint_check = const.CONSTRAINT_CHECK_FAILURE

        # sample a random goal and set it as a user goal in the following episode
        self.goal = self.__sample_goal()
        self.goal[const.REQUEST_SLOTS_KEY][self.ultimate_request_slot] = 'UNK'

        # after sampling a goal, the user can take the initial actions
        init_action = self.__sample_random_init_action()

        return init_action

    def step(self, agt_action):
        """
         Overrides the abstract method from the super class
        """
        logging.info('Calling `GORuleBasedUser` __response_step method')

        # we need to increase it for 2, counting for the agent response afterwards
        self.current_turn_nb += 2
        self.episode_over = False
        self.dialog_status = const.NO_OUTCOME_YET

        agt_diaiact = agt_action[const.DIA_ACT_KEY]

        if self.max_nb_turns > 0 and self.current_turn_nb > self.max_nb_turns:
            self.dialog_status = const.FAILED_DIALOG
            self.episode_over = True
            self.state[const.DIA_ACT_KEY] = const.CLOSING_DIA_ACT_KEY
        else:
            self.state[const.USER_STATE_HISTORY_SLOTS].update(self.state[const.USER_STATE_INFORM_SLOTS])
            self.state[const.USER_STATE_INFORM_SLOTS].clear()

            # based on the last agent action
            if agt_diaiact == const.INFORM_DIA_ACT_KEY:
                self.__response_inform(agt_action)

            elif agt_diaiact == const.REQUEST_DIA_ACT_KEY:
                self.__response_request(agt_action)

            elif agt_diaiact == const.CONFIRM_ANSWER_DIA_ACT_KEY:
                self.__response_confirm_answer(agt_action)

            elif agt_diaiact == const.CLOSING_DIA_ACT_KEY:
                self.__response_closing(agt_action)
                self.episode_over = True

            elif agt_diaiact == const.MULTIPLE_CHOICE_DIA_ACT_KEY:
                self.__response_multiple_choice(agt_action)

            elif agt_diaiact == const.THANKS_DIA_ACT_KEY:
                self.__response_thanks(agt_action)
                self.episode_over = True

        # create the next user action (not sure about this)
        next_usr_action = {}
        next_usr_action[const.DIA_ACT_KEY] = self.state[const.DIA_ACT_KEY]
        next_usr_action[const.INFORM_SLOTS_KEY] = self.state[const.USER_STATE_INFORM_SLOTS]
        next_usr_action[const.REQUEST_SLOTS_KEY] = self.state[const.USER_STATE_REQUEST_SLOTS]

        self.__log_user_action(next_usr_action)

        return next_usr_action, self.episode_over, self.dialog_status


class GOModelBasedUser(GOSimulatedUser):
    """
    Class representing a model based user in the Goal-Oriented Dialogue Systems.
    Extends the `GOUser` class.
    
    Class members:
    
        - ** model_path **: the path to save or load the model
    """

    def __init__(self, simulation_mode=None, goal_set=None, max_nb_turns=0, slot_set=None, act_set=None,
                 model_path=None):
        super(GOModelBasedUser, self).__init__(simulation_mode, goal_set, max_nb_turns, slot_set, act_set)

        logging.info('Calling `GOModelBasedUser` constructor')

        self.model_path = model_path

    def __log_user_goal(self, usr_goal):
        """
        Overrides the abstract method from the super class
        """

        # TODO
        raise NotImplementedError()

    def __log_user_action(self, usr_action):
        """
        Overrides the abstract method from the superclass
        """

        # TODO
        raise NotImplementedError()

    def __sample_random_init_action(self):
        # TODO
        raise NotImplementedError()

    def __sample_goal(self):
        # TODO
        raise NotImplementedError()

    def reset(self):
        """
        Overrides the abstract method from the super class
        """
        # TODO
        raise NotImplementedError()

    def step(self, agt_action):
        # TODO
        raise NotImplementedError()
