import torch.nn as nn
import torch.nn.functional as F
import torch

class inception(nn.Module):
    def __init__(self, input_channels, num_classes, use_1=False, step=1):
        super(inception, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_classes, (3, 3), stride=step, padding=1)
        self.conv2 = nn.Conv2d(num_classes, num_classes, (3, 3), stride=1, padding=1)
        if use_1:
            self.conv3 = nn.Conv2d(input_channels, num_classes, (1, 1), stride=step)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_classes)
        self.bn2 = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        if self.conv3:
            x = self.conv3(x)
        y = y + x
        return F.relu(y)

def resnet_block(input_channels, num_classes, num_inceptions, first=False):
    blk = []
    for i in range(num_inceptions):
        if i == 0 and not first:
            blk.append(inception(input_channels, num_classes, use_1=True, step=2))
        else:
            blk.append(inception(num_classes, num_classes, use_1=False, step=1))
    return blk

demo1 = nn.Sequential(nn.Conv2d(1, 64, (7, 7), 1, 3),
                      nn.BatchNorm2d(64),
                      nn.MaxPool2d((3, 3), (2, 2), (1, 1)))
demo2 = nn.Sequential(*resnet_block(64, 64, 2, True))
demo3 = nn.Sequential(*resnet_block(64, 128, 2, False))
demo4 = nn.Sequential(*resnet_block(128, 256, 2, False))
demo5 = nn.Sequential(*resnet_block(256, 512, 2, False))

ResNet = nn.Sequential(demo1, demo2, demo3, demo4, demo5,
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten(),
                       nn.Linear(512, 10))

if __name__ == '__main__':
    a = torch.randn(1, 1, 96, 96)
    b = ResNet(a)
    print(b.shape)
