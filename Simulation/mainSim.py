import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
from datetime import datetime

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

#device = "cpu"

# the transition model/states
Transition = namedtuple('Transition', ('state1', 'state2', 'action', 'next_state1', 'next_state2', 'reward'))

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
        self.layer1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        # second layer has 6 input channels and goes to 3 output channels
        self.layer2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        # second layer has 6 input channels and goes to 3 output channels
        self.layer3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        # first linear layer
        self.fc1 = nn.Linear(5947, 5832)
        # second linear layer
        self.fc2 = nn.Linear(5832, 2916)
        # third linear layer
        self.fc3 = nn.Linear(2916, 1458)
        # fourth linear layer
        self.fc4 = nn.Linear(1458, n_actions)

        
    def forward(self, x, y):
        x = self.layer1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.flatten(1)
        x = torch.cat((x, y), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        return x
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
# MEMORYSIZE is the amount of total replay memory used for batch sampling
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 25000
TAU = 0.005
LR = 1e-4
MEMORYSIZE = 1000

# number of actions
n_actions = env.actionSpace

# variables for recording
# list of episode lengths
episode_durations = []
# list of total rewards
total_rewards = []
# list of max phase
max_phase = []
end_phase = []
# list of time per phase
time_per_phase = []

policy_net = ArmDQN(n_actions).to(device)
target_net = ArmDQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(MEMORYSIZE)

steps_done = [ 0, 0, 0, 0]

# select which action to take next
# either us the policy_net output or get rand action
def select_action(state1, state2, phase):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done[phase]/ EPS_DECAY)
    steps_done[phase] += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state1, state2).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, (n_actions - 1))]], device=device, dtype=torch.long)
    

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
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state1)), device=device, dtype=torch.bool)
    non_final_next_states1 = torch.cat([s for s in batch.next_state1 if s is not None])
    non_final_next_states2 = torch.cat([s for s in batch.next_state2 if s is not None])
    state1_batch = torch.cat(batch.state1)
    state2_batch = torch.cat(batch.state2)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) with the model
    # these are the actions which would've been taken for each batch state according to policy
    state_action_values = policy_net(state1_batch, state2_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states1, non_final_next_states2).max(1).values
    
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
    # append total reward with 0 and add two it each step of the episode
    total_rewards.append(0)
    time_per_phase.append([0,0,0,0])
    max_phase.append(0)

    state1, state2 = env.resetSim()
    state1 = torch.tensor(state1, dtype=torch.float32, device=device).unsqueeze(0)
    state1 = state1.permute(0, 3, 2, 1)
    state2 = torch.tensor(state2, dtype=torch.float32, device=device).unsqueeze(0)
    
    print(f"Episode: {i_episodes}")
    
    for t in count():
        prev_phase = env.phase

        action = select_action(state1, state2, env.phase)
        observation, reward, terminated = env.takeAction(action.item())
        total_rewards[i_episodes] += reward
        reward = torch.tensor([reward], device=device)

        if terminated:
          next_state1 = None
          next_state2 = None
        else:
          next_state1 = torch.tensor(observation[0], dtype=torch.float32, device=device).unsqueeze(0)
          next_state1 = next_state1.permute(0, 3, 2, 1)
          next_state2 = torch.tensor(observation[1], dtype=torch.float32, device=device).unsqueeze(0)

        # store transition in memory
        memory.push(state1, state2, action, next_state1, next_state2, reward)

        # move to next state
        state1 = next_state1
        state2 = next_state2

        # optimize the model
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if prev_phase != env.phase:
            time_per_phase[i_episodes][prev_phase] += t + 1
        
        if prev_phase < env.phase:
            max_phase[i_episodes] = env.phase

        if terminated or t > 500:
            episode_durations.append(t + 1)
            end_phase.append(env.phase)
            time_per_phase[i_episodes][env.phase] = t + 1
            break


year = datetime.today()
time = datetime.now().time()
testId = "Test02"

formatted_year = year.strftime("%Y-%m-%d")
formatted_time = time.strftime("%H:%M:%S")

directory = "Models/" + formatted_year
if not os.path.exists(directory):
    os.mkdir(directory)

model_path = directory + "/" + testId + "_" + formatted_time + ".pth"

torch.save(policy_net.state_dict(), model_path)

directory = "Results/" + formatted_year
if not os.path.exists(directory):
    os.mkdir(directory)

testInfo_path = directory + "/" + testId + "_" + formatted_time + ".csv"

with open(testInfo_path, "w") as file:
    file.write("Test Information for " + testId)

with open(testInfo_path, "a") as file:
    file.write("\nbatch_size, Gamma, Eps_start, Eps_end, Eps_decay, Tau, Learning_rate, memory_size")
    file.write(f"\n{BATCH_SIZE}, {GAMMA}, {EPS_START}, {EPS_DECAY}, {TAU}, {LR}, {MEMORYSIZE}")
    file.write("\nepisode, total_rewards, duration, max_phase, end_phase, time_per_phase")
    for i in range(num_episodes):
        file.write(f"\n{i}, {total_rewards[i]}, {episode_durations[i]}, {max_phase[i]}, {end_phase[i]}, {time_per_phase[i]}")

print(f"Total rewards: {total_rewards}")
print(f"Episode durations: {episode_durations}")
print(f"Max phase: {max_phase}")
#print(f"Time per phase: {time_per_phase}")

env.disconnect()

