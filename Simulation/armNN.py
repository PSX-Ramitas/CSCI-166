import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        # dropout for convolution
        self.drop = nn.Dropout2d()
        # layer for switching to fully connected linear network
        # is set to none since input must be set dynamically based on the convolutional network
        # the input amount is correct for initial input of size [ 3, 480, 270]
        self.fc1 = nn.Linear(5947, 1024)
        # second linear layer
        self.fc2 = nn.Linear(1024, 729)
        # third linear layer
        self.fc3 = nn.Linear(729, n_actions)

        
    def forward(self, x, y):
        x = self.layer1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.drop(x)
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
        return x