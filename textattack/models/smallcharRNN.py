import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class smallcharRNN(nn.Module):
    def __init__(self, classes=4, bidirection = False, layernum=1, char_size = 69, hiddensize = 100):
        super(smallcharRNN, self).__init__()
        self.lstm = nn.LSTM(char_size, hiddensize, layernum, bidirectional = bidirection)
        self.hiddensize = hiddensize
        numdirections = 1 + bidirection
        self.hsize = numdirections * layernum
        self.linear = nn.Linear(hiddensize * numdirections, classes)
        self.log_softmax = nn.LogSoftmax()
    def forward(self, x):
        h0 = Variable(torch.zeros(self.hsize, x.size(0), self.hiddensize)).to(device)
        c0 = Variable(torch.zeros(self.hsize, x.size(0), self.hiddensize)).to(device)
        x = x.transpose(0,1)
        x = x.transpose(0,2)
        x,(hn,cn) = self.lstm(x,(h0,c0))
        x = x[-1]
        x = self.log_softmax(self.linear(x))
        return x