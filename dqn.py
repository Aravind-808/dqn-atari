'''
some basic terms to know before proceeding 

episode: 
an episode is one full game from start to finish. for eg. in subway surfers, 
its from when the guy starts running till when he crashes into something and gets caught 
by the policeman. good agents have higher total reward than bad ones, or shorter episodes than bad ones,
depending on the situation/nature of the task

step: 
a step is one "tick" of the game loop. agent sees screen, picks action,
game moves forward a frame, agent gets reward.

transition: 
the atomic unit of experience. basically a tuple of whatever happens 
in a step: (state, action, reward, next_state, done)

replay buffer: 
a circular queue of transitions with a fixed capacity.
every step adds one transition, when its full the oldest gets overwritten.
during training you sample random batches from it instead of learning
sequentially. the randomness breaks correlations so the network doesnt
overfit to whatever the agent is currently doing. basically the whole point
is to make the training data look random even though game frames are highly
correlated.

State:
what the agent sees. here its the last 4 grayscale 84x84 frames
stacked together so the agent can infer motion.

Action:
what the agent does. breakout has 4: nothing, fire, left, right.

Reward:
the score the environment gives after each step. +1 for breaking
a brick, 0 otherwise. the whole goal is to maximise total reward per episode.

Q-value - Q(state, action):
expected total future reward if you take this
action now and play optimally after. the network learns to predict this.

Policy:
the decision rule. given a state, which action to pick. here its
epsilon-greedy: usually pick the highest Q-value action, but with probability
epsilon pick randomly instead.

now the timing stuff that actually controls when things happen:

total_timesteps;
how long training runs. 2M is enough to see learning start,
10M is where you get real performance.

learning_starts:
the Q-network doesnt update at all for the first 80k steps.
the agent just acts randomly and fills the buffer. this matters because
Q-learning is off-policy - it can learn from any experience regardless of
how it was collected, so none of that early random data gets wasted. if you
started training at step 32 (one batch) youd be overfitting to the first
few seconds of one game.

train_frequency:
only train every 4 steps, not every single one. each step
only adds one new transition so training every step just re-learns the same
data.

target_update_freq:
every 10k steps copy the online network weights into
the target network. between copies the target stays frozen. without this the
bellman target y = r + gamma * max Q_target(s') shifts every single update
and training diverges - youre chasing a moving goalpost.
'''

import os
import random
import time

from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, List
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
)

ale_py.register_v5_envs()

@dataclass
class Config:
    env_id = "ALE/Breakout-v5"
    seed = 42

    # start_learning: 80,000 means for the first 80k steps,
    # do absolutely nothing. this is to fill the replay buffer with enough
    # transition data for the model to start learning so that it doesnt overfit

    # train freq means train only every 4 steps. each step adds just 1 transition to the buffer
    #

    # target_update_freq:
    # every 10k steps copy the online network weights into
    # the target network. between copies the target stays frozen. without this the
    # bellman target y = r + gamma * max Q_target(s') shifts every single update
    total_steps = 10_000_000
    start_learning = 80_000
    train_freq = 4
    target_update = 10_000

    batch_size = 32
    gamma = 0.99
    lr = 1e-4
    grad_clip = 10.0 # clipping grads to prevent large af gradient updates

    buffer_size = 100_000

    # eps is used to control exploration - exploitation
    # initially, as eps is higher, the agent will prefer exploration
    # as it anneals every 500k steps, it will move towards exploitation as
    # it will learn stuff
    eps_st = 1.0
    eps_end = 0.01
    eps_anneal = 500_000

    log_interval = 1_000
    save_interval = 100_000
    save_dir = "checkpoints"


class ReplayBuffer:
    # circular q(ueue) implementation 
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device

        self.pos = 0
        self.size = 0

        self.obs = np.zeros((capacity, *obs_shape), dtype = np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype = np.uint8)

        self.actions = np.zeros((capacity, ), dtype = np.uint8)
        self.rewards = np.zeros((capacity, ), dtype = np.float32)
        self.dones = np.zeros((capacity, ), dtype = np.float32)

    def push(self, obs, action, reward, next_obs, done):
        # store a single transition
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos= (self.pos + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)
    
    def sample(self, batch_size):
        # sample a random batch of transitions and return as tensors
        pass

