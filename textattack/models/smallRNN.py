import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
class smallRNN(nn.Module):
    def __init__(self, classes=4, bidirection = False, layernum=1, length=20000,embedding_size =100, hiddensize = 100):
        super(smallRNN, self).__init__()
        self.embd = nn.Embedding(length, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hiddensize, layernum, bidirectional = bidirection)
        self.hiddensize = hiddensize
        numdirections = 1 + bidirection
        self.hsize = numdirections * layernum
        self.linear = nn.Linear(hiddensize * numdirections, classes)
        self.log_softmax = nn.LogSoftmax()
    def forward(self, x, returnembd = False):
        embd = self.embd(x)
        if returnembd:
            embd = Variable(embd.data, requires_grad=True).cuda()
            embd.retain_grad()
        h0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).to(device)
        c0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).to(device)
        x = embd.transpose(0,1)
        x,(hn,cn) = self.lstm(x,(h0,c0))
        x = x[-1]
        x = self.log_softmax(self.linear(x))
        if returnembd:
            return embd,x
        else:
            return x