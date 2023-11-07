import torch
from torch import nn
import torch.nn.functional as F

class ST(nn.Module):
    def __init__(self, inplanes, planes, inshape, outshape):
        super(ST, self).__init__()
        self.inshape = inshape
        self.outshape = outshape
        self.conv1 = nn.Conv2d(inplanes, 20, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
       
        self.fc0 = nn.Linear(1 * 1 * 20, 20)
        self.fc = nn.Linear(20, 6)


    def localization(self, x):

        xs = self.conv1(x)
        xs = self.relu(xs)
        xs = self.conv2(xs)
        xs = self.relu(xs)

        xs = self.avgpool(xs)
        xs = xs.view(xs.size(0), -1)
        xs = self.fc0(xs)
        xs = self.relu(xs)
        xs = self.fc(xs)

        return xs


    def st(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, torch.Size([x.size(0), x.size(1), self.outshape, self.outshape]))
        x = F.grid_sample(x, grid)

        return x


    def forward(self, x):
        out = self.st(x)
        # out = x + xs
        # out = (out - torch.mean(out))/torch.std(out)

        return out
