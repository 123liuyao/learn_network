import torch.nn as nn
import torch


class AlxNet(nn.Module):
    def __init__(self):
        super(AlxNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=5*5*256, out_features=4096),
            nn.Linear(in_features=4096, out_features=1028),
            nn.Linear(in_features=1028, out_features=100)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x



if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    model = AlxNet()
    output = model(img)
    print(output, '\n', output.shape)
