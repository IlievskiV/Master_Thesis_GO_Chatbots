# TC-Bot-Restaurant

This system is training a Goal-Oriented Chatbot in the Restaurant Booking domain, such that the knowledge is trnsfered from the Movie Booking Domain.

In order to obtain results, run the script **src/domain_adaptation**.
All parameters for training the system are set and well documented
in the script.

## Explanation of the script

This script is training one bot from scartch and one bot with the transfered knowledge.

### Training and Testing process

The purpose of the Transfer Learning technique is to prove that with less data we can achieve better performance. For this reason, the script is randonly partitioning the dataset of user goals in the following portions: 5, 10, 20, 30, 50  and all 120. Then, using the user goal portions it trains both bots.

After the training process, both bots are tested on a set of 32 user goals.

### Results

All of the results are written in the folder **deep_dialog/checkpoints/results**. The results are organized as following:

- *training_records_pretrain.json*: for each iteration it contains the training success rate, total accumulated reward and total number of turns for the model with transfer learning

- *training_records_scratch.json*: for each iteration it contains the testing success rate, total accumulated reward and total number of turns for the model without transfer learning

- *testing_records_pretrain.json*: for each iteration it contains the testing success rate, total accumulated reward and total number of turns for the model with transfer learning

- *testing_records_scratch.json*: for each iteration it contains the testing success rate, total accumulated reward and total number of turns for the model without transfer learning

- *full_pretrain_performances.json*: for each iteration it contains the training performances for every epoch while training.

- *full_scratch_performances.json*: for each iteration it contains the testing performances (success rate, reward, number of turns) for every epoch while training.

### Data

The data for training and testing is placed in the folder **src/deep_dialog/data**. There are 120 training user goals in the file *user_goals.p* and 32 testing user goals in the file *user_test_goals*. The Knowledge Base entries are stored in the file *rest_kb*