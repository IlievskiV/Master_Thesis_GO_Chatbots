<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/dst/state_tracker.py#L13)</span>
### GOStateTracker

```python
core.dst.state_tracker.GOStateTracker(act_set=None, slot_set=None, max_nb_turns=None)
```


Abstract Base Class of all state trackers in the Goal-Oriented Dialogue Systems.

__Class members:__


- ** history **: list of both user and agent actions, such that they are in alternating order
- ** act_set **: the set of all intents used in the dialogue.
- ** slot_set **: the set of all slots used in the dialogue.
- ** act_set_cardinality **: the cardinality of the act set.
- ** slot_set_cardinality **: the cardinality of the slot set.
- ** current_slots **: a dictionary that keeps a running record of which slots are filled 
		(inform slots) and which are requested (request slots)
- ** state_dim **: the dimensionality of the state. It is calculated afterwards.
- ** max_nb_turns **: the maximal number of dialogue turns

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/dst/state_tracker.py#L133)</span>
### GORuleBasedStateTracker

```python
core.dst.state_tracker.GORuleBasedStateTracker(act_set=None, slot_set=None, max_nb_turns=None)
```


Class for Rule-Based state tracker in the Goal-Oriented Dialogue Systems.
Extends the `GOStateTracker` class.

__Class members:__


- ** state_dim **: the dimension of the state

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/dst/state_tracker.py#L427)</span>
### GOModelBasedStateTracker

```python
core.dst.state_tracker.GOModelBasedStateTracker(act_set=None, slot_set=None, max_nb_turns=None, is_training=None, model_path=None)
```


Class for Model-Based state tracker in the Goal-Oriented Dialogue Systems.
Extends the `GOStateTracker` class.

__Class members:__


- ** is_training **: boolean flag indicating the mode of using the model-based state tracker
- ** model_path **: the path to save or load the model

