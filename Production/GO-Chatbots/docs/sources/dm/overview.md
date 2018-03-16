<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/dm/dialogue_system.py#L13)</span>
### GODialogSys

```python
core.dm.dialogue_system.GODialogSys(act_set=None, slot_set=None, agt_feasible_actions=None, params=None)
```


The GO Dialogue System mediates the interaction between the environment and the agent.

__Class members: __


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
- ** kb_path **: path to any knowledge base
- ** agt_feasible_actions **: list of templates described as dictionaries, corresponding to each action the agent might take
			(dict to be specified)
- ** max_nb_turns **: the maximal number of dialogue turns


