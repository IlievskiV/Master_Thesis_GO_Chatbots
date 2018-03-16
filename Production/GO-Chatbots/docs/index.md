# Documentation for the Goal-Oriented Chatbots

In these pages, the documentation of the library for developing goal-oriented chatbots is provided. The goal-oriented chatbots are closed domain chatbots with a predetermined intent to help users complete a task. For example, the task could be a flight booking or restaurant reservation.

This library is modeling the goal-oriented chatbots as a Partially Observable Markov Decision Process (POMDP) using the Reinforcement Learning (RL) set of algorithms. Therefore, there is an agent, acting in a dialogue environment and learning the dialogue policy by receiving reward signals.

Since reinforcement learners require an environment to interact, conventional dialogue corpora cannot be used directly. For this reason, we need a simulator which will simulate the user behavior. Using this approach, on one side we have a simulated user with a predetermined goal to acheve, using the agent which is not aware of the user goal.

Both of them are generating actions, the simulated user is generating the action based on its goal, while the agent is generating the action based on its beliefs learned from the training so far.

## The notion of user goal

As we mentioned above, in this kind of chatbot scenario, the users are having goal. Therefore, we need to provide a formal goal definition. When the users are having goal, on one side they know some piece of information and on the other side they are searching for some other piece of information that they would like to know.

These chunks of information, which are the point of interest, are called *slots*, such that the slot is having a *key name* (some general term) and a *value*, i.e. they act as a key-value pairs. Following the user's nature, we define two types of slots: ** inform ** and ** request ** slots. The ** inform ** slots have value, while the ** request ** slots do not have value. The slots are domain dependent and differ from domain to domain. Therefore, they are specified in the input, as it is explained in the [Input](#Input) section.

For example, in a restaurant booking scenario, the user might know the type of cuisine, for example *Indian*, but she might not know any restaurant serving that kind of food. Therefore, we will have an ** inform ** slot named *cuisine* with a value *Indian* and one ** request ** slot named *restaurant*. The chatbot should give an answer to that.

In this system, the user goal is encoded in the following semantic structure:


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *{* <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *"request_slots" : { ... }*, <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *"inform_slots" : { ... }* <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *}*


## User and agent actions

As we already mentioned, both, the agent and the user are taking actions. The user action is derived from its goal, the agent action is derived from its policy. In order to establish a convention in the system, the actions are all having same format. It is including the following info: type of the action, inform slots, request slots and the natural language representation.

The type of action, or the dialogue act (intent) is describing the nature of the action. Since, for now there in only a rule based user simulator, the system supports 11 dialogue acts, which are the following: 

- request
- inform
- confirm_question
- confirm_answer
- greeting
- closing
- multiple_choice
- thanks
- welcome
- deny
- not_sure

The inform and request slots are same as in the goal, while the natural language representation is a sencente corresponding to the the action.



