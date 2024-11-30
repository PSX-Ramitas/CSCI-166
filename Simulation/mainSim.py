import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import armSim

# load connect to 
env = armSim.ArmEnv(False) # set to false to disable the gui
env.connect()   # start the pybullet server
env.load_environment()  #load the environment

# setup which device to use
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# the transition model/states
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# class to hold the replay memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# create a class for the nueral network model
class ArmDQN(nn.Module):

    def __init__(self, n_actions):
        super(ArmDQN, self).__init__()
        # first layer goes from 3 input channels to 6 output channels
        self.layer1 = nn.Conv2d(3, 3, kernel_size=3)
        # second layer has 6 input channels and goes to 3 output channels
        self.layer2 = nn.Conv2d(3, 3, kernel_size=3)
        # dropout for convolution
        self.drop = nn.Dropout2d()
        # layer for switching to fully connected linear network
        # is set to none since input must be set dynamically based on the convolutional network
        # the input amount is correct for initial input of size [ 3, 480, 270]
        self.fc1 = nn.Linear(23364, 19683)
        # second linear layer
        self.fc2 = nn.Linear(19683, 6561)
        # third linear layer
        self.fc3 = nn.Linear(6561, 2187)
        # fourth linear layer to output
        self.fc4 = nn.Linear(2187, 729)

        
    def forward(self, x):
        x = self.layer1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# number of actions
n_actions = env.actionSpace

print(device)

policy_net = ArmDQN(n_actions).to(device)
target_net = ArmDQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

# select which action to take next
# either us the policy_net output or get rand action
def select_action(state1, state2):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done/ EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state1).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, n_actions)]], device=device, dtype=torch.long)
    
episode_durations = []

def optimize_model():
    # check if enouch action state pairs in memory
    if len(memory) < BATCH_SIZE:
        return
    
    # get a sample from memory of batch size
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) with the model
    # these are the actions which would've been taken for each batch state according to policy
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    #Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Hubber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 1
else:
    num_episodes = 1

for i_episodes in range(num_episodes):
    # Initialize the environment and get its state
    print(device)

    state1, state2 = env.resetSim()
    state1 = torch.tensor(state1, dtype=torch.float32, device=device).unsqueeze(0)
    state1 = state1.permute(0, 3, 2, 1)
    #state2 = torch.tensor(state2, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        action = policy_net(state1).max(1).indices.view(1, 1)
        print(action)

