
# python main2.py --model myrits --epochs 100 --batch_size 64 --out short_out --data patdata/json_short --runname shorttry1 --cv 0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from ipdb import set_trace
from sklearn import metrics

SEQ_LEN = 28
PARAM_LEN = 28

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        # torch.eye(): Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self, batch_size=64, lr=0.001, rnn_hid_size=50, num_classes=5, impute_weight=1, label_weight=1, lambda_reg=0.0001, alpha_reg=0.9, drop_out=0.25, loss_weights=[1,1,1,1,1]):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.lr = lr
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg
        self.drop_out = drop_out
        self.loss_weights = loss_weights

        self.build()
        
    def set_params(self, batch_size, lr, rnn_hid_size, num_classes, impute_weight, label_weight, lambda_reg, alpha_reg, drop_out, loss_weights):
        self.__init__(
             batch_size, lr, rnn_hid_size, num_classes, impute_weight, label_weight, lambda_reg, alpha_reg, drop_out, loss_weights
        )
        return self
    
    def get_params(self, deep=True):
        return {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'rnn_hid_size': self.rnn_hid_size,
            'num_classes': self.num_classes,
            'impute_weight': self.impute_weight,
            'label_weight': self.label_weight,
            'lambda_reg': self.lambda_reg,
            'alpha_reg': self.alpha_reg,
            'drop_out': self.drop_out,
            'loss_weights': self.loss_weights,
        }
    
    
    def build(self):
#         print(self.rnn_hid_size)
        self.rnn_cell = nn.LSTMCell(PARAM_LEN*2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size = PARAM_LEN, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = PARAM_LEN, output_size = PARAM_LEN, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, PARAM_LEN)
        self.feat_reg = FeatureRegression(PARAM_LEN)

        self.weight_combine = nn.Linear(PARAM_LEN * 2, PARAM_LEN)

        self.dropout = nn.Dropout(p = self.drop_out)
        self.out = nn.Linear(self.rnn_hid_size, self.num_classes)

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        #labels = data['labels'].view(-1, 1)
        labels = data['labels']

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        h = self.dropout(h)       
        y_h = self.out(h)
        y_h = torch.sigmoid(y_h)
        #print(y_h)
        pos_weight = torch.FloatTensor(self.loss_weights).cuda()
#         criterion = nn.BCEWithLogitsLoss(weight=pos_weight)
        # criterion = nn.CrossEntropyLoss(weight=pos_weight)
        # labels = torch.LongTensor(labels)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        y_loss = criterion(y_h, labels)
        #print(y_loss)
        l1_norm = torch.norm(self.out.weight, p=1).cuda()
        l2_norm = torch.norm(self.out.weight).cuda()
        y_loss += self.lambda_reg * ((1-self.alpha_reg)*l2_norm + self.alpha_reg*l1_norm)

        
        return {'loss': x_loss * self.impute_weight + y_loss * self.label_weight, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'h': h,\
                'evals': evals, 'eval_masks': eval_masks,'xloss': x_loss, 'yloss': y_loss}

    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
