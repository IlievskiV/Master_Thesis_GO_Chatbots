<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/agent/agents.py#L10)</span>
### GODQNAgent

```python
core.agent.agents.GODQNAgent()
```

 Class for the Goal-Oriented agents with a DQN-based policy learning.

This class is extending the class AbstractDQNAgent from keras-rl framework.
This type of DQN agent should learn the policies of a dialogue in a given environment.

One agent action is represented as a dictionary having the exact same structure as the user action, which is:

- ** diaact **: the act (intent) of the action
- ** inform_slots **: the set of informed slots
- ** request_slots **: the set of request slots
- ** nl **: the natural language representation of the agent action

The following methods are implemented:

- `forward`
- `backward`
- `compile`
- `load_weights`
- `save_weights`
- `layers`


__Class members:__


From  `Agent` class:

- ** processor ** (`Processor` instance): glue between the environment and the agent

From  `AbstractDQNAgent` class:

- ** nb_actions **: the number of all possible actions
- ** memory ** (`Memory` instance): the type of buffer the agent will use. For example: SequentialMemory
- ** gamma **: the discount reward factor. Default is 0.99, the agent remembers everything
- ** batch_size **: the number of memories to replay in one training epoch. Default is 32
- ** nb_steps_warmup **: the number of steps needed to warm up and fill the memory replay buffer. Default is 1000
- ** train_interval **: no idea
- ** memory_interval **: no idea
- ** target_model_update **: after how many steps, the target DQN gets updated. Default is 10 000
- ** delta_range **: no idea. Default is None. This variable is deprecated
- ** delta_clip **: no idea. Default is Inf
- ** custom_model_objects **: no idea

From `DQNAgent` class:

- ** policy ** (`Policy` instance): The policy that the agent follows. Default is None
- ** test_policy ** (`Policy` instance): The policy that the agent follows during testing. Default is None
- ** enable_double_dqn **: To enable the Double DQN technique. By default is on
- ** enable_dueling_network **: To enble the Dueling DQN technique. Be default is off
- ** dueling_type **: If the dueling is on, what type of it. The default is average

