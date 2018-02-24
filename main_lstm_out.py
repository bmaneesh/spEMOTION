import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torch.autograd import Variable
import torch.optim as optim
from utils import batch
import nltk
import random
import torch.nn.functional as F
from sklearn_crfsuite import metrics
# from utils import getBatch

cuda = torch.cuda.is_available()
gpus = [2]
if cuda:
    torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor

all_data = np.load('iemocap_sixclass.npz')
data = all_data['data']
data = np.reshape(data,(data.shape[0], data.shape[1], 1))
targets = all_data['label']
class_num = targets.max() + 1
# targets_onehot = np.zeros((targets.shape[0], class_num))
targets_onehot = np.reshape(targets, (targets.shape[0], 1))
# class_num = 1
# for sample in enumerate(targets):
#     targets_onehot[sample[0], sample[1]] = 1
data_shape = data.shape
# Normalize
# data = (data - data.mean())/data.std()
print data.shape, targets_onehot.shape

class lstm_att(nn.Module):
    def __init__(self, input_size,num_layers, hidden_size, drop_rate, dense_out, align_hidd,class_num):
        super(lstm_att,self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, num_layers = num_layers, hidden_size = hidden_size
            , bidirectional=True, dropout = drop_rate, batch_first = True)
        self.dense = nn.Linear(hidden_size*2,1)
        self.dense_out = nn.Linear(hidden_size*2,class_num)
        self.soft = nn.Softmax()
        self.dropout = nn.Dropout(drop_rate)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # # n`= hidden units in align. model
        # # n = hidden units in  model OR hidden layer size??
        # self.Wa = Variable(torch.randn(1,align_hidd, hidden_size)).repeat(1000,1,1) # n`*n;here n->2n?
        # self.Ua = Variable(torch.randn(1,align_hidd, 2*hidden_size)).repeat(1000,1,1) # n`*2n
        # self.Va = Variable(torch.randn(1,align_hidd,1)).repeat(1000,1,1) # n`

    def forward(self, input, istraining=False):
        # print input.size()
        lstm_out, hidden = self.lstm(input) #https://github.com/pytorch/pytorch/issues/3587
        # print lstm_out.size()
        # drop_out = self.dropout(lstm_out)
        alpha = self.atten(lstm_out)
        # print alpha.size(), hidden[0].size()
        # contxt = torch.matmul(alpha.transpose_(1,2), lstm_out)
        contxt = torch.matmul(alpha.view(alpha.size()[0], 1, alpha.size()[1]), lstm_out)
        # print contxt.size()
        out = torch.squeeze(self.sigmoid(self.dense_out(contxt)))
        # print out
        return out

    def atten(self, hid):
        # print 'hidden shape:{0}'.format(hid.size())
        U = self.dense(hid)
        # print 'U-shape:{0}'.format(U.size())
        alpha = self.soft(self.tanh(U).view(U.size()[0], U.size()[1]))
        return alpha

LEARNING_RATE = 0.01
NEPOCHS = 1
BATCH_SIZE = 16

# input = Variable(torch.randn(1000,70000,1))#1000 batch size, 100 seq len
inputs = torch.from_numpy(data).type(FloatTensor)
targets = Variable(torch.from_numpy(targets_onehot).type(LongTensor))
# print inputs.type, targets.type
model = lstm_att(input_size = 1,num_layers = 1, hidden_size = 20, drop_rate = 0.3, dense_out=10, align_hidd=10,class_num=class_num)
print model.parameters()#, inputs[0*BATCH_SIZE:(0+1)*BATCH_SIZE,].shape

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
mean = [-0.5329]
std = [1302.4]# torch.std(input)
norm = Normalize(mean, std)
inputs = Variable(norm(inputs.permute(2,0,1)).permute(1,2,0))
# print mean, std

if cuda:
    model = model.cuda()
    # inputs = inputs.cuda()
    # targets = targets.cuda()
    print 'GPU enabled'

for epoch in range(NEPOCHS):
    losses = []
    print 'epoch-{0}/{1}'.format(epoch, NEPOCHS)
    for bid, bind in enumerate(batch(data_shape[0], BATCH_SIZE)):
        binputs, btargets = inputs[bind,:,:], targets[bind,:]
        # binputs, btargets = inputs[bind,:,:], targets[bind]
        if cuda:
            binputs.cuda()
            btargets.cuda()
        pred = model(binputs, istraining = True)
        model.zero_grad()
        loss = loss_function(pred, torch.squeeze(btargets))
        losses.append(loss.data.tolist()[0])
        loss.backward()
        optimizer.step()
        if bid%1000 == 0:
            print 'batch-{0} loss-{1}'.format(bid, sum(losses)/len(losses))
