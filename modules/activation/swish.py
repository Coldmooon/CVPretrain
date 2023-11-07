from torch import nn
import torch

class Swish(nn.Module):
    def __init__(self, beta=1, inplace=False, channels=None):
        super(Swish, self).__init__()
        c = (channels == None) and 1 or int(channels)
        # self.beta = nn.Parameter(torch.ones([1,c,1,1])) # .fill_(beta)
        self.inplace = inplace
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        if self.inplace:
            x.mul_(self.sigmoid(3.5 * x))
            return x
        else:
            return x * self.sigmoid(3.5 * x)


class RecSwish(nn.Module):
    def __init__(self, beta=1, inplace=False, channels=None):
        super(RecSwish, self).__init__()
        c = (channels == None) and 1 or int(channels)
        # self.beta = nn.Parameter(torch.ones([1,c,1,1])) # .fill_(beta)
        self.inplace = inplace
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        if self.inplace:
            x[x<0] = x[x<0] * self.sigmoid(3.5*x[x<0])
            return x
        else:
            t = x.clone() 
            t[t<0] = x[x<0] * self.sigmoid(3.5*x[x<0])
            return t
