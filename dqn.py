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
    total_steps = 2_000_000
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
    eps_anneal = 1_000_000

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

        self.actions = np.zeros((capacity, ), dtype = np.int64)
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
        idxs = np.random.randint(0, self.size, size = batch_size)

        # normalize since stored as uint
        obs = torch.tensor(self.obs[idxs], dtype = torch.float32,device=self.device) / 255.0
        next_obs = torch.tensor(self.next_obs[idxs], dtype = torch.float32,device=self.device) / 255.0

        actions = torch.tensor(self.actions[idxs], dtype = torch.int64,device = self.device)
        rewards = torch.tensor(self.rewards[idxs], dtype = torch.float32, device = self.device)
        dones= torch.as_tensor(self.dones[idxs],dtype=torch.float32, device=self.device)
        
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return self.size

'''
input: stack of 4 grayscale frames: dims = 84x84 (4, 84, 84)
output: q value for every action (n_actions, )

Conv(32, 8x8, stride 4), Conv(64, 4x4, stride 2)
Conv(64, 3x3, stride 1), FC(512) → FC(n_actions)
'''
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        # out_size = floor((input_size - kernel_size) / stride) + 1
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))

def select_action(qnet, obs, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)

    else:
        obs_t = torch.as_tensor(obs, dtype = torch.float32, device = device).unsqueeze(0)/255.0
        with torch.no_grad():
            q_vals = qnet(obs_t)
        
        return int(q_vals.argmax(dim = 1).item())

def linear_epsilon(step, cfg):
    # linearly decay epsilon value from eps_start to eps_end
    fraction = min(step / cfg.eps_anneal, 1.0)
    return cfg.eps_st + fraction * (cfg.eps_end - cfg.eps_st)

# emulate an target net (separate from online net) 
# that updates only every 10k steps
def compute_loss(online_net, target_net, batch, gamma):
    obs, actions, rewards, next_obs, dones = batch
    
    # current q_vals from the action taken
    q_vals = online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        max_next_q = target_net(next_obs).max(dim=1).values
        targets = rewards + gamma * max_next_q * (1.0 - dones)
    
    # choosing huber loss over MSE
    loss = nn.functional.smooth_l1_loss(q_vals, targets)
    return loss

def make_env(env_id, seed, render = False):
    render_mode = "human" if render else None

    no_skip_id = env_id.replace("-v5", "NoFrameskip-v4").replace("ALE/", "")
    env = gym.make(no_skip_id, render_mode=render_mode)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4)
    env = FrameStackObservation(env, stack_size=4)
    env.action_space.seed(seed)
    return env

def train(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg.env_id, cfg.seed)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (4, 84, 84)
    print(f"Env: {cfg.env_id}  |  Actions: {n_actions}  |  Obs: {obs_shape}")

    online_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, obs_shape, device)

    # METRICS
    ep_rewards = []
    ep_lens = []
    losses = []

    obs, _ = env.reset(seed=cfg.seed)
    ep_reward, ep_len = 0.0, 0
    start_time = time.time()

    print("STARTING TRAINING !!!!!!")

    '''
    for reference 
    total_steps = 10_000_000
    start_learning = 80_000
    train_freq = 4
    target_update = 10_000
    '''

    for step in range(1, cfg.total_steps+1):
        eps = linear_epsilon(step, cfg)
        action = select_action(online_net, obs, eps, n_actions, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
        clipped_rew = float(np.clip(reward, -1.0, 1.0))

        buffer.push(obs, action, clipped_rew, next_obs, terminated)
        obs = next_obs
        ep_reward+=reward
        ep_len += 1

        if done:
            ep_rewards.append(ep_reward)
            ep_lens.append(ep_len)

            obs, _ = env.reset()
            ep_reward, ep_len = 0.0,0

        if step >= cfg.start_learning and step % cfg.train_freq == 0:
            batch = buffer.sample(cfg.batch_size)
            loss  = compute_loss(online_net, target_net, batch, cfg.gamma)
 
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), cfg.grad_clip)
            optimizer.step()
 
            losses.append(loss.item())
        
        if step % cfg.target_update == 0:
            target_net.load_state_dict(online_net.state_dict())
        
        if step % cfg.log_interval == 0:
            sps = step / (time.time() - start_time)
            n_ep = len(ep_rewards)
            recent = ep_rewards[-20:] if n_ep else [0]
            mean_r = np.mean(recent)
            mean_l = np.mean(losses[-100:]) if losses else float("nan")
 
            print(
                f"step={step:>8,}  eps={eps:.3f}  "
                f"ep={n_ep:>6,}  mean_reward(20)={mean_r:>7.1f}  "
                f"loss={mean_l:.4f}  buf={len(buffer):>7,}  sps={sps:.0f}"
            )

        if step % cfg.save_interval == 0:
            path = os.path.join(cfg.save_dir, f"dqn_{step}.pt")
            torch.save({
                "step":       step,
                "online":     online_net.state_dict(),
                "target":     target_net.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "ep_rewards": ep_rewards,
            }, path)
            print(f"  → checkpoint saved: {path}")
    
    env.close()

    final_path = os.path.join(cfg.save_dir, "best_model.pt")
    torch.save({
        "step":       cfg.total_steps,
        "online":     online_net.state_dict(),
        "target":     target_net.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "ep_rewards": ep_rewards,
        "config":     cfg,
    }, final_path)
    print(f"\nTraining complete. Final model saved at {final_path}")
    return online_net, ep_rewards

if __name__ == "__main__":
    cfg = Config()
    train(cfg)