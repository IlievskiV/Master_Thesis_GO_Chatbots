"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the Goal-Oriented Dialogue agent classes.
"""
from keras.models import Sequential
from keras.layers import Dense, Activation

from rl.agents.dqn import DQNAgent
import logging

from core import constants as const

class GODQNAgent():

    def __init__(self, output_dim=0, state_dimension=0, hidden_size=80, act_func=const.RELU, *args, **kwargs):

        logging.info("Calling `GODQNAgent` constructor")
        self.output_dim = output_dim
        self.state_dimension = state_dimension
        self.hidden_size = hidden_size
        self.act_func = act_func

        # build the model
        model = self.__build_model()

        # Log the summary if the model
        logging.info(model.to_json())

        super(GODQNAgent, self).__init__(model=model, *args, **kwargs)

    def __build_model(self):
        """
        Private helper method to build the agent Neural Net Model
        """
        logging.info("Calling `GODQNAgent` build_model method")

        model = Sequential()

        # Hidden layer
        model.add(Dense(self.hidden_size, input_shape=(self.state_dimension,)))
        model.add(Activation(self.act_func))

        # Output layer
        model.add(Dense(self.output_dim))
        model.add(Activation(const.LINEAR))

        return model






