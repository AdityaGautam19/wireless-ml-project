import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 3)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.fc1 = nn.Linear(32*124, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.classifier(x), self.regressor(x)