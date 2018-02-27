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
from sklearn import metrics
# from utils import getBatch

cuda = torch.cuda.is_available()
gpus = [2]
bi_dir = 1 # if bidirectional
train_frac = 0.7 # training splir
if cuda:
    torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor

fac = 2 if bi_dir else 1
# all_data = np.load('iemocap_sixclass.npz')
# data = all_data['data']
# data = np.reshape(data,(data.shape[0], data.shape[1], 1))
all_data = np.load('spect_MEAN.npz')
data = all_data['data']
targets = all_data['label']
class_num = targets.max() + 1
# targets_onehot = np.zeros((targets.shape[0], class_num))
targets_onehot = np.reshape(targets, (targets.shape[0], 1))
# class_num = 1
# for sample in enumerate(targets):
#     targets_onehot[sample[0], sample[1]] = 1
# Normalize
# data = (data - data.mean())/data.std()
print data.shape, targets_onehot.shape

class lstm_att(nn.Module):
    def __init__(self, input_size,num_layers, hidden_size, drop_rate, dense_out, align_hidd,class_num, factor):
        super(lstm_att,self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, num_layers = num_layers, hidden_size = hidden_size
            , bidirectional=True, dropout = drop_rate, batch_first = True)
        self.dense = nn.Linear(hidden_size*factor,1)
        self.dense_out = nn.Linear(hidden_size*factor,class_num)
        self.soft = nn.Softmax()
        # self.dropout = nn.Dropout(drop_rate)
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

LEARNING_RATE = 0.001
NEPOCHS = 500
BATCH_SIZE = 32

# input = Variable(torch.randn(1000,70000,1))#1000 batch size, 100 seq_len
print data.dtype
inputs = torch.from_numpy(np.abs(data)).type(FloatTensor)
targets = Variable(torch.from_numpy(targets_onehot).type(LongTensor))
# print inputs.type, targets.type


mean = [-0.5329]
std = [1302.4]# torch.std(input)
norm = Normalize(mean, std)
inputs = Variable(norm(inputs.permute(2,0,1)).permute(1,2,0))
data_shape = inputs.size()
rand_order = range(0,data_shape[0])
random.shuffle(rand_order)
print len(rand_order[0:np.floor(data_shape[0]*train_frac).astype('int32')])
train_data = inputs[rand_order[0:np.floor(data_shape[0]*train_frac).astype('int32')], :, :]
test_data = inputs[rand_order[np.ceil(data_shape[0]*train_frac).astype('int32'):], :, :]
train_label = targets[rand_order[0:np.floor(data_shape[0]*train_frac).astype('int32')], :]
test_label = targets[rand_order[np.ceil(data_shape[0]*train_frac).astype('int32'):], :].data.cpu().numpy()
# variable->np error as variable stores history of the object and np has no provision
# variable.data-> tensor->.numpy() gives array this can only be done on CPU so use .cpu()
data_shape = train_data.size()


model = lstm_att(input_size = inputs.size()[-1],num_layers = 2, hidden_size = 20,
                 factor = fac, drop_rate = 0.3, dense_out=10, align_hidd=10,class_num=class_num)
print model#, inputs[0*BATCH_SIZE:(0+1)*BATCH_SIZE,].shape
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE,)

if cuda:
    model = model.cuda()
    train_data = train_data.cuda()
    test_data = test_data.cuda()
    train_label = train_label.cuda()
    # test_label = test_label.cuda()
    print 'GPU enabled'

for epoch in range(NEPOCHS):
    losses = []
    print 'epoch-{0}/{1}'.format(epoch, NEPOCHS)
    for bid, bind in enumerate(batch(data_shape[0], BATCH_SIZE)):
        binputs, btargets = train_data[bind,:,:], train_label[bind,:]
        # if cuda:
        #     binputs.cuda()
        #     btargets.cuda()
        pred = model(binputs, istraining = True)
        model.zero_grad()
        loss = loss_function(pred, torch.squeeze(btargets))
        losses.append(loss.data.tolist()[0])
        loss.backward()
        optimizer.step()
        if bid%10 == 0:
            print 'batch-{0} loss-{1}'.format(bid, sum(losses)/len(losses))

test_pred = model(test_data, istraining = False)
values, test_cls_indices = torch.max(test_pred, dim = 1)
print metrics.classification_report(test_label, test_cls_indices.data.cpu().numpy(), digits=3)