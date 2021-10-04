import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
    
        self.RELUConvBN0 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.RELUConvBN1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.RELUConvBN2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.RELUConvBN3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.RELUConvBN4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.shortcut = nn.Sequential()
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        node1 = self.RELUConvBN0(x)
        node2 = self.RELUConvBN2(node1) + self.RELUConvBN1(x)
        out = self.RELUConvBN3(node2) + self.RELUConvBN4(node1) + self.shortcut(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes):
            super(ResBlock, self).__init__()
            self.conv_a = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )
            self.conv_b = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )
            self.down_sample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
            )


    def forward(self, x):
        out = self.conv_b(self.conv_a(x)) + self.down_sample(x)
        return out

class MINININ(nn.Module):
    def __init__(self, planes, num_classes=10):
        super(MINININ, self).__init__()
        self.planes = planes
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8)
        )
        self.cell0 = BasicBlock(8, planes[0])
        self.cell1 = ResBlock(planes[0], planes[1])
        self.cell2 = BasicBlock(planes[1],planes[2])
        self.cell3 = ResBlock(planes[2],planes[3])
        self.cell4 = BasicBlock(planes[3],planes[4])
        self.last_act = nn.Sequential(
            nn.BatchNorm2d(planes[4]),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(in_features = planes[4], out_features = num_classes, bias=True)

    def forward(self, x):
        out = self.stem(x)
        out = self.cell0(out)
        out = self.cell1(out)
        out = self.cell2(out)
        out = self.cell3(out)
        out = self.cell4(out)
        out = self.last_act(out)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def nin(planes, num_classes=10):
    planes = list(planes)
    planes = np.asarray([int(plane) for plane in planes])
    planes = 40+planes*8
    return MINININ(planes, num_classes)