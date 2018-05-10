#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import libraries
import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from matplotlib import pyplot as plt
<<<<<<< HEAD:dqn.py
from memory_dqn import Replay_Memory
=======
from replay_buffer import PrioritizedReplayBuffer
>>>>>>> 05b51165adabb7c8b484dc84e605d6375a23e737:dqn_her.py

# bit flipping environment
class Env():
    def __init__(self, size = 8, shaped_reward = False):
        # number of bits in env
        self.size = size

        # whether to use shaped reward or not
        self.shaped_reward = shaped_reward

        # initialize state bit array
        self.state = np.random.randint(2, size = size)

        # generate target bit array to be achieved
        self.target = np.random.randint(2, size = size)
        while np.sum(self.state == self.target) == size:
            self.target = np.random.randint(2, size = size)

    def step(self, action):
        # modify bit at action index
        self.state[action] = 1 - self.state[action]

        # return reward for operation
        if self.shaped_reward:
            # use shaped reward which is more informative
            return np.copy(self.state), -np.sum(np.square(self.state - self.target))
        else:
            # use binary reward
            if not np.sum(self.state == self.target) == self.size:
                return np.copy(self.state), -1
            else:
                return np.copy(self.state), 0

    def reset(self, size = None):
        # initialize state and target variables at end of episode
        if size is None:
            size = self.size
        self.state = np.random.randint(2, size = size)
        self.target = np.random.randint(2, size = size)

<<<<<<< HEAD:dqn.py
# Experience Memory Buffer wrapper class
class MemoryBuffer(object):
    def __init__(self,
                 buffer_size = 50000,
                 burn_in = 10000,
                 combined = False,
                 prioritized = False,
                 hindsight = False,
                 default_goal = None,
                 priority_alpha = None):
        # wrap a replay memory object
        self.buffer_size = buffer_size
        self.memory = Replay_Memory(memory_size = buffer_size,
                                    burn_in = burn_in,
                                    combined = combined,
                                    prioritized = prioritized,
                                    hindsight = hindsight,
                                    default_goal = default_goal,
                                    priority_alpha = priority_alpha)

    def add(self, state, action, reward, next_state, goal):
        # add experience to memory
        # note that the transition must be wrapped in a tuple
        transition = (state.copy(), action, reward, next_state.copy(), goal)
        self.memory.append(transition)

    def sample(self, size, beta = 0.5):
        # encodes a sample from priorization of memory buffer
        batch, weights, indices = self.memory.sample_batch(size, beta)
        states, actions, rewards, next_states, goals = zip(*batch)
        return states, actions, rewards, next_states, goals, weights, indices
=======
# Experience Replay Buffer with priority
class Buffer(object):

    def __init__(self, buffer_size = 50000, alpha = 0.5):
        # initialize buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, goal):
        # add experience to buffer
        self.buffer.add(obs_t = state,
                        action = action,
                        reward = reward,
                        obs_tp1 = next_state,
                        done = goal)

    def sample(self, size, beta=0.7):
        # encode sample from PrioritizedReplayBuffer
        return self.buffer.sample(size, beta)
>>>>>>> 05b51165adabb7c8b484dc84e605d6375a23e737:dqn_her.py

# utility function for fully connected layer
def fully_connected_layer(inputs, dim, activation = None, scope = "fc", reuse = None, init = tf.contrib.layers.xavier_initializer(), bias = True):
    with tf.variable_scope(scope, reuse = reuse):
        w_ = tf.get_variable("W_", [inputs.shape[-1], dim], initializer = init)
        outputs = tf.matmul(inputs, w_)
        if bias:
            b = tf.get_variable("b_", dim, initializer = tf.zeros_initializer())
            outputs += b
        if activation is not None:
            outputs = activation(outputs)
        return outputs

# 1 Layer Feed Forward Network used for Q-network
class Model():
    def __init__(self, size, name):
        # create network
        with tf.variable_scope(name):
            self.size = size

            # input place holder
            self.inputs = tf.placeholder(shape = [None, self.size * 2], dtype = tf.float32)

            # fully connected layer
            init = tf.contrib.layers.variance_scaling_initializer(factor = 1.0, mode = "FAN_AVG", uniform = False)
            self.hidden = fully_connected_layer(self.inputs, 256, activation = tf.nn.relu, init = init, scope = "fc")

            # output Q layer
            self.Q_ = fully_connected_layer(self.hidden, self.size, activation = None, scope = "Q", bias = False)
            
            # perform argmax on Q-values to return action
            self.predict = tf.argmax(self.Q_, axis = -1)

            # place holder for action performed
            self.action = tf.placeholder(shape = None, dtype = tf.int32)
            self.action_onehot = tf.one_hot(self.action, self.size, dtype = tf.float32)
            
            # compute estimated Q value
            self.Q = tf.reduce_sum(tf.multiply(self.Q_, self.action_onehot), axis = 1)
            
            # place holder for Q_{t+1}
            self.Q_next = tf.placeholder(shape=None, dtype=tf.float32)
            
            # define loss function for network
            self.loss = tf.reduce_sum(tf.square(self.Q_next - self.Q))
            
            # initialize optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = self.optimizer.minimize(self.loss)

            # variable initializer
            self.init_op = tf.global_variables_initializer()

# function to update target graph
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*(1. - tau)) + (tau * tfVars[idx+total_vars//2].value())))
    return op_holder

# utility function to run target update session
def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, required=True, help='experiment name')

    parser.add_argument('--her', action='store_true', default=False, help='flag for using HER or not')
    parser.add_argument('--shaping', action='store_true', default=False, help='flag to use reward shaping')
    parser.add_argument('--test', action='store_false', default=True, help='flag to whether only test')

    parser.add_argument('-s', '--size', type=int, default=15, help='number of bits in environment')
    parser.add_argument('-k', '--ratio', type=int, default=4, help='ratio of HER buffer to reg buffer')

    args = parser.parse_args()

    # variables to decide mode
    HER = args.her
    shaped_reward = args.shaping
    
    # number of bits in environment
    size = args.size
    
    # K value for K-future strategy
    K = args.ratio
    
    # variables for training Q-network
    num_epochs = 5
    num_cycles = 50
    batch_size = 128
    num_episodes = 16
    optimisation_steps = 40
    
    # variables to initialize Q-network
    tau = 0.95
    gamma = 0.98
    epsilon = 0.0
    buffer_size = 1e6

    # buffers to track progress
    succeed = 0
    total_loss = []
    success_rate = []
    total_rewards = []

    # flags to save/train model
    train = args.test
    save_model = True

    # create directory to save model
    model_dir = args.experiment+'/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # initialize Q-networks
    modelNetwork = Model(size = size, name = "model")
    targetNetwork = Model(size = size, name = "target")

    # operations for target graph
    trainables = tf.trainable_variables()
    updateOps = updateTargetGraph(trainables, tau)

    # initialize environment and experience buffer
    # you can add other arguments here
    buff = MemoryBuffer(buffer_size)
    env = Env(size = size, shaped_reward = shaped_reward)
    
    # train the network
    if train:

        # generate figure to show statistics
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(211)
        plt.title("Success Rate")
        ax.set_ylim([0,1.])
        ax2 = fig.add_subplot(212)
        plt.title("Q Loss")
        line = ax.plot(np.zeros(1), np.zeros(1), 'b-')[0]
        line2 = ax2.plot(np.zeros(1), np.zeros(1), 'b-')[0]
        fig.canvas.draw()

        # start training session
        with tf.Session() as sess:
            # initialize network variables
            sess.run(modelNetwork.init_op)
            sess.run(targetNetwork.init_op)

            # loop for num_epochs
            for i in tqdm(range(num_epochs), total = num_epochs):

                # loop for num_cycles
                for j in range(num_cycles):
                    # track network performance in each cycle
                    successes = []
                    total_reward = 0.0

                    # loop for num_episodes
                    for n in range(num_episodes):
                        # reset environment
                        env.reset()

                        # initialize episode buffers
                        episode_experience = []
                        episode_succeeded = False

                        # run episode for size timesteps
                        for t in range(size):
                            # get state and target
                            s = np.copy(env.state)
                            g = np.copy(env.target)

                            # use both as input to network
                            inputs = np.concatenate([s,g],axis = -1)

                            # obtain predicted action from network
                            action = sess.run(modelNetwork.predict,feed_dict = {modelNetwork.inputs:[inputs]})
                            action = action[0]
                            
                            # perform epsilon-greedy policy
                            if np.random.rand(1) < epsilon:
                                action = np.random.randint(size)
                            s_next, reward = env.step(action)

                            # append to episode experience
                            episode_experience.append((s,action,reward,s_next,g))
                            total_reward += reward
                            
                            # check success for episode
                            if reward == 0:
                                if episode_succeeded:
                                    continue
                                else:
                                    episode_succeeded = True
                                    succeed += 1

                        # store episode result to successes
                        successes.append(episode_succeeded)
                        
                        # append to experience buffer
                        for t in range(size):
                            # generate buffer for deuling q-network
                            s, a, r, s_n, g = episode_experience[t]
                            buff.add(s, a, r, s_n, g)
                            # inputs = np.concatenate([s,g],axis = -1)
                            # new_inputs = np.concatenate([s_n,g],axis = -1)
                            # buff.add(np.reshape(np.array([inputs,a,r,new_inputs]),[1,4]))
                            
                            # add additional information for HER
                            if HER:
                                # for every sample from regular experience add 4 from HER
                                for k in range(K):
                                    # pick a random time step from future
                                    ###############
                                    # what happens when we change this strategy?
                                    ###############
                                    future = np.random.randint(t, size)

                                    # obtain new goal for experience
                                    _, _, _, g_n, _ = episode_experience[future]

                                    # generate experience from new goal
                                    inputs = np.concatenate([s,g_n],axis = -1)
                                    new_inputs = np.concatenate([s_n, g_n],axis = -1)
                                    final = np.sum(np.array(s_n) == np.array(g_n)) == size
                                    if shaped_reward:
                                        r_n = 0 if final else -np.sum(np.square(np.array(s_n) == np.array(g_n)))
                                    else:
                                        r_n = 0 if final else -1

                                    # add new experience to experience buffer
                                    buff.add(s, a, r_n, s_n, g_n)
<<<<<<< HEAD:dqn.py
=======
                                    # buff.add(np.reshape(np.array([inputs,a,r_n,new_inputs]),[1,4]))
>>>>>>> 05b51165adabb7c8b484dc84e605d6375a23e737:dqn_her.py

                    # train the Q-network once every cycle
                    mean_loss = []

                    # perform optimization_steps updates per cycle
                    for k in range(optimisation_steps):
                        
                        # obtain a batch from experience buffer
                        s, a, r, s_next, g, _, _ = buff.sample(batch_size)
                        s = np.dstack((s, g))
                        s_next = np.dstack((s_next, g))
                        s = np.reshape(s, (batch_size, size * 2))
                        s_next = np.reshape(s_next, (batch_size, size * 2))

                        # generate predicted Q-values
                        Q1 = sess.run(modelNetwork.Q_, feed_dict = {modelNetwork.inputs: s_next})

                        # generate target Q-values from target network
                        Q2 = sess.run(targetNetwork.Q_, feed_dict = {targetNetwork.inputs: s_next})
                        doubleQ = Q2[:, np.argmax(Q1, axis = -1)]
                        Q_target = np.clip(r + gamma * doubleQ,  -1. / (1 - gamma), 0)

                        # obtain loss and update network
                        _, loss = sess.run([modelNetwork.train_op, modelNetwork.loss], feed_dict = {modelNetwork.inputs: s, modelNetwork.Q_next: Q_target, modelNetwork.action: a})

                        # append loss to mean_loss
                        mean_loss.append(loss)

                    # update target network every cycle
                    updateTarget(updateOps,sess)

                    # update buffers every cycle
                    total_rewards.append(total_reward)
                    total_loss.append(np.mean(mean_loss))
                    success_rate.append(np.mean(successes))
                    
                    # update plot for every cycle
                    ax.relim()
                    ax.autoscale_view()
                    ax2.relim()
                    ax2.autoscale_view()
                    line.set_data(np.arange(len(success_rate)), np.array(success_rate))
                    line2.set_data(np.arange(len(total_loss)), np.array(total_loss))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(1e-7)
            
            # save model to file in every epoch
            if save_model:
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(model_dir, "model.ckpt"))

        # indicator at end of training
        print("Number of episodes succeeded: {}".format(succeed))
        input("Press enter...")
    
    # evaluate trained network
    with tf.Session() as sess:
        # load trained model from file
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_dir, "model.ckpt"))

        # loop over network until termination
        while True:
            # reset environment
            env.reset()

            # print initial state and target
            print("Initial State:\t{}".format(env.state))
            print("Goal:\t{}".format(env.target))

            # run Q-network for the episode
            for t in range(size):
                # obtain current state and goal
                s = np.copy(env.state)
                g = np.copy(env.target)

                # provide inputs
                inputs = np.concatenate([s,g],axis = -1)

                # obtain predicted action
                action = sess.run(targetNetwork.predict,feed_dict = {targetNetwork.inputs:[inputs]})
                action = action[0]

                # obtain next state and reward
                s_next, reward = env.step(action)
                print("State at step {}: {}".format(t, env.state))
                if reward == 0:
                    print("Success!")
                    break

            # prompt when episode ends
            input("Press enter...")

if __name__ == "__main__":
    main()
