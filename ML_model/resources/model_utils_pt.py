import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._nb_classes_ = 15

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(1, 64, 5, padding = (2,2)) # add padding to keep input and output same size
        nn.init.xavier_normal(self.conv1.weight)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(64, 128, 5, padding = (2,2))

        # Calculate input size for the first FC layer
        x = torch.randn(80,80).view(-1,1,80,80)
        self._to_linear = None
        self.convs(x)

        # FC layer 1
        self.fc1 = nn.Linear(self._to_linear, 50)

        # FC layer 2
        self.fc2 = nn.Linear(50, self._nb_classes_)

        # Initialize weights
        self.weight_init()

    # initalize weights
    def weight_init(self):
        for m in self.children():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_normal_(m.weight) # AKA Glorot normal initialization
            if m.bias.data is not None:
                m.bias.data.fill_(0)


    # Forward pass through convolutional layers
    def convs(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(10, 10))
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(8, 8))
        x = F.relu(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    # Forward pass through all layers
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x)
