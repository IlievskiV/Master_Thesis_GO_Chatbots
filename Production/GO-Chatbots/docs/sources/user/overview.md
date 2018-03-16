<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/user/users.py#L11)</span>
### GOUser

```python
core.user.users.GOUser(id=None, simulation_mode=None, goal_set=None)
```


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

__Class members:__


- ** id **: the id of the user
- ** current_turn_nb **: the number of the current dialogue turn
- ** state **: user internal state, keeping record of the past and current actions
- ** simulation_mode **: semantic frame or natural language sentence form of user utterances
- ** goal_set **: the set of goals for the user
- ** goal **: the user goal in the current dialogue turn

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/user/users.py#L100)</span>
### GORealUser

```python
core.user.users.GORealUser(id=None, goal_set=None)
```


Class connecting a real user, writing on the standard input. Extends the `GOUser` class.

__Class members:__


----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/user/users.py#L141)</span>
### GOSimulatedUser

```python
core.user.users.GOSimulatedUser(id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None)
```


Abstract Base Class for all simulated users in the Goal-Oriented Dialogue Systems.
Extends the `GOUser` class.

__Class members:__


- ** slot_set **: the set of all slots in the dialogue scenario
- ** act_set **: the set of all acts (intents) in the dialogue scenario
- ** dialog_status **: the status of the dialogue from the user perspective. The user is deciding whether the
		   dialogue is finished or not. The dialogue status could have the following value:

	- ** NO_OUTCOME_YET **: the dialogue is still ongoing
	- ** SUCCESS_DIALOG **: the dialogue was successful, i.e. the user achieved the goal
	- ** FAILED_DIALOG **: the dialogue failed, i.e. the user didn't succeed to achieve the goal

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/user/users.py#L235)</span>
### GORuleBasedUser

```python
core.user.users.GORuleBasedUser(id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None, init_inform_slots=None, ultimate_request_slot=None)
```


Abstract class representing a rule-based user in the Goal-Oriented Dialogue Systems.
Since, it is a rule-based simulated user, it will be domain-specific.
Extends the `GOUser` class.

Class members:

- ** init_inform_slots **: list of initial inform slots, such that if the current user goal is containing some
			   of them, they must appear in the initial user turn
			   
- ** ultimate_request_slot ** : the slot that is the actual goal of the user, and everything is around this slot.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/user/users.py#L897)</span>
### GOModelBasedUser

```python
core.user.users.GOModelBasedUser(id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None, is_training=None, model_path=None)
```


Class representing a model based user in the Goal-Oriented Dialogue Systems.
Extends the `GOUser` class.

Class members:

- ** is_training **: boolean flag indicating the mode of using te model-based state tracker
- ** model_path **: the path to save or load the model

