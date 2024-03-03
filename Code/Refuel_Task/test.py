import torch
from torch import nn


class YourModel(nn.Module):
    def __init__(self, d):
        super(YourModel, self).__init__()
        self.percept = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, d),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.percept(x)
        print("Shape after percept:", x.shape)
        return x


model = YourModel(128)
x = torch.randn(1, 6, 2)
output = model(x)
print("Shape after model:", output.shape)