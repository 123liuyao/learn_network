import torch.nn as nn
import torch
from torch.nn import functional as F

class inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(inception, self).__init__(**kwargs)
        # 线路一，单1*1卷积层
        self.c1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路二，1*1和3*3卷积
        self.c2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.c2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路三，1*1和5*5卷积
        self.c3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.c3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路四，3*3最大池化， 1*1卷积
        self.c4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.c4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.c1(x))
        p2 = F.relu(self.c2_2(F.relu(self.c2_1(x))))
        p3 = F.relu(self.c3_2(F.relu(self.c3_1(x))))
        p4 = F.relu(self.c4_2(self.c4_1(x)))
        return torch.cat((p1, p2, p3, p4), 1)

class googlenet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(googlenet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            inception(192, 64, [96, 128], [16, 32], 32),
            inception(256, 128, [128, 192], [32, 96], 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            inception(480, 192, [96, 208], [16, 48], 64),
            inception(512, 160, [112, 224], [24, 64], 64),
            inception(512, 128, [128, 256], [24, 64], 64),
            inception(512, 112, [144, 288], [32, 64], 64),
            inception(528, 256, [160, 320], [32, 128], 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            inception(832, 256, [160, 320], [32, 128], 128),
            inception(832, 384, [192, 384], [48, 128], 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(p=0.1),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = googlenet(in_channels=1, num_classes=10)
    a = torch.randn(1, 1, 96, 96)
    b = model(a)
    print(b.shape)
