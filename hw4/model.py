import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3,128,3)
        self.conv2 = nn.Conv2d(128,128,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*31*31,1000)
        self.fc2 = nn.Linear(1000,10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 128*31*31)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3,128,3)
        self.conv2 = nn.Conv2d(128,128,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*14*14,1000)
        self.fc2 = nn.Linear(1000,10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128*14*14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3,128,3, padding=1)
        self.conv2 = nn.Conv2d(128,128,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*15*15,1000)
        self.fc2 = nn.Linear(1000,10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128*15*15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x