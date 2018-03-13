import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torch.nn.modules.module import _addindent
from torch.autograd import Variable
import torch.optim as optim
from utils import batch
import random
from sklearn import metrics
from tensorboardX import SummaryWriter

writer = SummaryWriter()
cuda = torch.cuda.is_available()
gpus = [0]
bi_dir = 1 # if bidirectional
split_frac = [0.7, 0.1, 0.2] # train-val-test split
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




data = np.delete(data, np.where(targets==5), axis=0)# remove surprise
targets = np.delete(targets, np.where(targets==5))
# data = np.delete(data, np.where(targets==2), axis=0)# remove frus.
# targets = np.delete(targets, np.where(targets==2))
# temp = np.where(targets>2)
# targets[temp] = targets[temp]-1

class_num = targets.max() + 1
# print np.unique(targets), np.histogram(targets, class_num)[0], data.shape


# targets_onehot = np.zeros((targets.shape[0], class_num))
targets_onehot = np.reshape(targets, (targets.shape[0], 1))
# class_num = 1
# for sample in enumerate(targets):
#     targets_onehot[sample[0], sample[1]] = 1
# Normalize
# data = (data - data.mean())/data.std()
dist = []
dist_norm = []
cls_wt = np.zeros(targets.shape)
for cls in np.unique(targets):
    dist.append(np.where(targets == cls)[0].shape[0])
    cls_wt[np.where(targets == cls)[0]] = 1.0/dist[-1]
    dist_norm.append(float(targets.shape[0])/dist[-1])
cls_wt = np.reshape(np.array(cls_wt),[cls_wt.shape[0],1])
# print data.shape, targets_onehot.shape, dist_norm, cls_wt.shape



class lstm_att(nn.Module):
    def __init__(self, input_size,num_layers, hidden_size, drop_rate, dense_out, align_hidd,class_num, factor):
        super(lstm_att,self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, num_layers = num_layers, hidden_size = hidden_size
            , bidirectional=True, dropout = drop_rate, batch_first = True)
        # self.dense = nn.Linear(hidden_size*factor, 1)
        self.dense = nn.Linear(hidden_size * factor, align_hidd)
        self.dense2 = nn.Linear(align_hidd,1)
        #####
        self.dense_out = nn.Linear(hidden_size * factor, dense_out)
        self.dense_class = nn.Linear(dense_out, class_num)
        #####
        self.soft = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(drop_rate)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # # n`= hidden units in align. model
        # # n = hidden units in  model OR hidden layer size??
        # self.Wa = Variable(torch.randn(1,align_hidd, hidden_size)).repeat(1000,1,1) # n`*n;here n->2n?
        # self.Ua = Variable(torch.randn(1,align_hidd, 2*hidden_size)).repeat(1000,1,1) # n`*2n
        # self.Va = Variable(torch.randn(1,align_hidd,1)).repeat(1000,1,1) # n`

    def forward(self, input, istraining=False):
        # print input.size()
        self.lstm.flatten_parameters()
        lstm_out, hidden = self.lstm(input) #https://github.com/pytorch/pytorch/issues/3587
        # print lstm_out.size()
        # drop_out = self.dropout(lstm_out)
        alpha = self.atten(lstm_out, istraining)
        # print 'alpha-{0} hidden-{1}'.format(alpha.size(), hidden[0].size())
        # contxt = torch.matmul(alpha.transpose_(1,2), lstm_out)
        contxt = torch.matmul(alpha.view(alpha.size()[0], 1, alpha.size()[1]), lstm_out)
        # print 'context size-{0}'.format(contxt.size())
        # out = torch.squeeze(self.sigmoid(self.dense_out(contxt)))
        #####
        # contxt_out = self.sigmoid(self.dense_out(contxt))
        # contxt_out = self.dropout(contxt_out)
        # out = torch.squeeze(self.sigmoid(self.dense_class(contxt_out)))
        contxt_out = self.relu(self.dense_out(contxt))
        if istraining:
            contxt_out = self.dropout(contxt_out)
        out = torch.squeeze(self.tanh(self.dense_class(contxt_out)))
        #####
        # print 'out size-{0}'.format(out.size())
        return out

    def atten(self, hid, istraining):
        # print 'hidden shape:{0}'.format(hid.size())
        ## U = self.dense(hid)
        # print 'U-shape:{0}'.format(U.size())
        ## alpha = self.soft(self.tanh(U).view(U.size()[0], U.size()[1]))
        U = self.relu(self.dense(hid))
        if istraining:
            U = self.dropout(U)
        U = self.relu(self.dense2(U))
        alpha = self.soft(self.relu(U).view(U.size()[0], U.size()[1]))
        return alpha


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr






LEARNING_RATE = 0.01
NEPOCHS = 30
BATCH_SIZE = 32
HIDDEN_SIZE = 32
NUM_LAYERS = 1
inputs = torch.from_numpy(np.abs(data)).type(FloatTensor)
targets = Variable(torch.from_numpy(targets_onehot).type(LongTensor))





mean = [torch.mean(inputs)]# [-0.5329]
std = [torch.std(inputs)]# [1302.4]# torch.std(inputs)
norm = Normalize(mean, std)
inputs = Variable(norm(inputs.permute(2,0,1)).permute(1,2,0), requires_grad = True)
print inputs.requires_grad
data_shape = inputs.size()
rand_order = range(0, data_shape[0])
random.Random(1729).shuffle(rand_order)
# print len(rand_order[0:np.floor(data_shape[0]*split_frac[0]).astype('int32')])
train_data = inputs[rand_order[0:np.floor(data_shape[0]*split_frac[0]).astype('int32')], :, :]
val_data = inputs[rand_order[np.ceil(data_shape[0]*split_frac[0]).astype('int32'):np.floor(data_shape[0]*sum(split_frac[0:2])).astype('int32')], :, :]
test_data = inputs[rand_order[np.ceil(data_shape[0]*sum(split_frac[0:2])).astype('int32'):], :, :]
train_label = targets[rand_order[0:np.floor(data_shape[0]*split_frac[0]).astype('int32')], :]
val_label = targets[rand_order[np.ceil(data_shape[0]*split_frac[0]).astype('int32'):np.floor(data_shape[0]*sum(split_frac[0:2])).astype('int32')], :]
test_label = targets[rand_order[np.ceil(data_shape[0]*sum(split_frac[0:2])).astype('int32'):], :]
# variable->np error as variable stores history of the object and np has no provision
# variable.data-> tensor->.numpy() gives array this can only be done on CPU so use .cpu()
train_data_shape = train_data.size()
# print np.histogram(train_label.data.cpu().numpy(), class_num)[0], np.histogram(test_label.data.cpu().numpy(), class_num)[0], np.unique(train_label.data.cpu().numpy())
# print train_data.size(), val_data.size(), test_data.size(), train_label.size(), val_label.size(), test_label.size()







model = lstm_att(input_size = inputs.size()[-1],num_layers = NUM_LAYERS, hidden_size = HIDDEN_SIZE,
                 factor = fac, drop_rate = 0.1, dense_out = 32, align_hidd=50,class_num=class_num)
# print model
# v, i = torch.max(train_label, dim=1)
# print np.unique(v.data.cpu().numpy())






loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(dist_norm)).type(FloatTensor))
# loss_function = nn.NLLLoss(weight=torch.from_numpy(np.array(dist_norm)).type(FloatTensor))
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min', factor=0.1, verbose=True, patience=2)

# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model, device_ids=[0, 1, 2]) # multi-GPU implementation trial





if cuda:
    model = model.cuda()
    train_data = train_data.cuda()
    test_data = test_data.cuda()
    val_data = val_data.cuda()
    val_label = val_label.cuda()
    train_label = train_label.cuda()
    test_label = test_label.cuda()







for name, param in model.named_parameters():
    if 'lstm' and 'weight' in name:
        nn.init.kaiming_normal(param)
        # print name+' initialized with norm. dist.'
    elif 'dense' and 'weight' in name:
        nn.init.kaiming_normal(param)
        # print name + ' initialized with xavier norm. dist.'
    elif 'dense' and 'bias' in name:
        nn.init.constant(param, 0)
        # print name + ' initialized with const. zero.'
    # print name








for epoch in range(NEPOCHS):
    losses = []
    # print 'epoch-{0}/{1}'.format(epoch, NEPOCHS)
    for bid, bind in enumerate(batch(train_data_shape[0], BATCH_SIZE)):
        binputs, btargets = train_data[bind,:,:], train_label[bind,:]
        pred = model(binputs, istraining = True)
        model.zero_grad()
        loss = loss_function(pred, torch.squeeze(btargets))
        losses.append(loss) if cuda else losses.append(loss.data.tolist()[0])
        loss.backward()
        optimizer.step()
        # TODO look for gradients
        print 'gradient-{0}'.format(inputs.grad)
        # if bid%10 == 0:
        #     print 'batch-{:d} loss-{:4f}'.format(bid, sum(losses)/len(losses))
    # print 'epoch-{0}/{1} train loss-{:.4f}'.format(epoch, NEPOCHS,sum(losses)/len(losses))
    # if epoch%10 == 0:
    val_pred = model(val_data, istraining = False)
    val_loss = loss_function(val_pred, torch.squeeze(val_label))
    values, val_cls_indices = torch.max(val_pred, dim = 1)
    print 'epoch-{:2d}/{:d} train_loss-{:.4f} val_loss-{:.4f} weighted val accuracy-{:.4f}'.format(epoch, NEPOCHS, sum(losses)/len(losses), val_loss.data
                                                                                                   , metrics.accuracy_score(np.squeeze(val_label.data.cpu().numpy()), np.squeeze(val_cls_indices.data.cpu().numpy()),
                                                         sample_weight=np.squeeze(cls_wt[rand_order[np.ceil(data_shape[0]*split_frac[0]).astype('int32'):np.floor(data_shape[0]*sum(split_frac[0:2])).astype('int32')],:])))
    # print metrics.classification_report(val_label.data.cpu().numpy(), val_cls_indices.data.cpu().numpy(), digits=3)
    lr_scheduler.step(val_loss)
    writer.add_scalar('data/scalar1', sum(losses)/len(losses), epoch)
    writer.add_scalar('data/scalar2', val_loss, epoch)
    # print 'weighted val accuracy-{:.4f}'.format(metrics.accuracy_score(np.squeeze(val_label.data.cpu().numpy()), np.squeeze(val_cls_indices.data.cpu().numpy()),
    #                                                      sample_weight=np.squeeze(cls_wt[rand_order[np.ceil(data_shape[0]*split_frac[0]).astype('int32'):np.floor(data_shape[0]*sum(split_frac[0:2])).astype('int32')],:])))














test_pred = model(test_data, istraining=False)
test_loss = loss_function(test_pred, torch.squeeze(test_label))
values, test_cls_indices = torch.max(test_pred, dim=1)
# print test_cls_indices
print 'test loss-{:.4f}\n'.format(test_loss)
print metrics.classification_report(test_label.data.cpu().numpy(), test_cls_indices.data.cpu().numpy(),
                                    digits=3)
print 'weighted test accuracy-{0}'.format(metrics.accuracy_score(np.squeeze(test_label.data.cpu().numpy()), np.squeeze(test_cls_indices.data.cpu().numpy()),
                                                                 sample_weight=np.squeeze(cls_wt[rand_order[np.ceil(data_shape[0]*sum(split_frac[0:2])).astype('int32'):],:])))

print metrics.confusion_matrix(test_label.data.cpu().numpy(), test_cls_indices.data.cpu().numpy())

writer.export_scalars_to_json("./all_scalars.json")
writer.close()