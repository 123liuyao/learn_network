import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), padding=(2, 2)),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, (5, 5), padding=(2, 2)),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == '__main__':
    input = torch.randn(1, 1, 28, 28)
    model = LeNet()
    output = model(input)
    print(output.shape)
