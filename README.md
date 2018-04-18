# Hindsight Experience Replay
[Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf) - Bit flipping experiment and Chase experiment in Tensorflow. 

Bit Flipping implementation includes:
* Double DQN with 1 hidden layer of size 256.
* Hindsight experience replay memory with "K-future" strategy.
* A very simple bit-flipping evironment as mentioned in the original paper.

Chase Experiment includes:
* DDPG Actor-Critic implementation.
* Hindsight experience replay memory with "K-future" strategy.
* A very simple reacher environment with continuous actions.

## Instructions

To run this code, adjust the hyperparameters from HER.py and type
```shell
$ python dqn_her.py -h
```
Read about the arguments provided in the code to experiment with different options.

## TODO

For bit flipping experiment
- [ ] Evaluate baseline and her for 15 bits env.
- [ ] Evaluate performance of baseline for different sizes (5-25).
- [ ] Evaluate performance of her for different sizes (5-25).
- [ ] Modify K-future strategy to final, K-episode, K-random strategies. 
- [ ] Evaluate performance for different strategies.

For chase experiment
- [ ] Understand implementation of DDPG
- [ ] Check similarities and dissimilarities with DQN
- [ ] Evaluate performance for different parameter values.

## Reference

DQN+HER: Mostly based on implementation by [minsangkim142](https://github.com/minsangkim142/hindsight-experience-replay).
DDPG+HER: Mostly based on implementation by [kwea123](https://github.com/kwea123/hindsight-experience-replay).