# module load python_gpu/3.7.1
from hyperopt import hp, tpe, fmin, Trials
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from models import myrits
import utils
import sys 
import hyperopt.pyll.stochastic
import numpy as np
import data_loader
import torch.optim as optim
from pytorchtools import EarlyStopping
import time
import argparse
import os
import sys

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import utils
import models
from models import myrits
import argparse
import data_loader
import pandas as pd
import ujson as ujson
import json

import time

from sklearn import metrics

from ipdb import set_trace

from time import gmtime, strftime
from cross_validation import k_folds

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str)
parser.add_argument('--runname', type=str)
parser.add_argument('--savepath', type=str, default='result')
parser.add_argument('--max_eval', type=int, default=20)
parser.add_argument('--equalweights', action='store_true')


args = parser.parse_args()

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)


datapath = args.data

MAX_EVALS = 500
out_file = './{}/{}.csv'.format(args.savepath, args.runname)
ITERATION = 0

of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['loss', 'params','epoch','iteration','train_time'])
of_connection.close()

'''
    Input: numpy array of label counts
    Returns: discounted weight array based on counts
'''
#TODO: OTHER WEIGHT EQUATIONS to try: 
def get_label_weight_ratio(counts):
    content = open('{}'.format(data)).readlines()
    labels = []
    for item in content:
        d = ujson.loads(item)
        labels.append(d['label'])
    labels = np.array(labels)
    labe = np.array([np.where(r==1)[0][0] for r in labels])
    unique, counts = np.unique(labe, return_counts=True)
    bettercounts = np.sum(labels, axis=0)
    return np.round(max(bettercounts)/bettercounts).tolist()    

def train(model, data_iter, optimizer, epoch):
    model.train()
    yloss = 0.0
    labels = []
    preds = []
    evals = []
    imputations = []
    save_impute = []
    save_label = []

    for idx, data in enumerate(data_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, optimizer, epoch)
        yloss += ret['yloss'].item() 

        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        
        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)
    preds = 1*(preds > 0.5)

    report = metrics.classification_report(labels, preds)
    file = open('./{}/{}_run{}_train_report'.format(args.savepath, args.runname, ITERATION), "w")
    file.write(report)
    file.write("\n")
    file.close()
    save_confusion = metrics.multilabel_confusion_matrix(labels, preds)
    np.save('./{}/{}_run{}_train_conf'.format(args.savepath, args.runname,ITERATION), save_confusion)


def evaluate(model, val_iter, epoch, early_stopping, ITERATION):
    model.eval()

    labels = []
    preds = []
    evals = []
    imputations = []
    save_impute = []
    save_label = []
    
    val_yloss = 0.0
    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        val_yloss += ret['yloss'].item() 

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        
        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)
    preds = 1*(preds > 0.5)

    # early stopping based on yloss
    average_val_loss = val_yloss/len(val_iter)
    early_stopping(average_val_loss, model)

    report = metrics.classification_report(labels, preds)
    file = open('./{}/{}_run{}_val_report'.format(args.savepath, args.runname, ITERATION), "w")
    file.write(report)
    file.write("\n")
    file.close()
    save_confusion = metrics.multilabel_confusion_matrix(labels, preds)
    np.save('./{}/{}_run{}_val_conf'.format(args.savepath, args.runname,ITERATION), save_confusion)
    
    return early_stopping.early_stop, average_val_loss

'''
    Split data into train and validation sets
    num_samples:
    train_ratio: 
'''
def train_val_split(num_samples, train_ratio, batch_size):
    if train_ratio > 1 or train_ratio <= 0:
        print('Training set ratio has to be smaller than 1, readjusted to 0.8.')
        train_ratio = 0.8
    indices = np.arange(num_samples)
    val_idx = np.random.choice(indices, int(num_samples*(1-train_ratio)))
    train_idx = np.setdiff1d(indices, val_idx)
    data_train, label_count = data_loader.get_loader(filename=datapath, indices=train_idx, batch_size=batch_size, get_labels=True)
    data_val = data_loader.get_loader(filename=datapath, indices=val_idx, batch_size=batch_size)  
    return data_train, data_val

def train_val_indices(num_samples, train_ratio):
    if train_ratio > 1 or train_ratio <= 0:
        print('Training set ratio has to be smaller than 1, readjusted to 0.8.')
        train_ratio = 0.8
    indices = np.arange(num_samples)
    val_idx = np.random.choice(indices, int(num_samples*(1-train_ratio)))
    train_idx = np.setdiff1d(indices, val_idx)
    return train_idx, val_idx

def run():
    
    space = {
            'batch_size': hp.choice('batch_size', [16,32,64]),
            'lr': hp.loguniform('lr', np.log(0.00005), np.log(0.001)),
            'rnn_hid_size': hp.quniform('rnn_hid_size', 50, 100, 10),
            'impute_weight': hp.quniform('impute_weight', 1, 1, 1),
            'label_weight': hp.quniform('label_weight', 1, 20, 1),
            'lambda_reg': hp.loguniform('lambda_reg', np.log(0.00005), np.log(0.001)),
            'alpha_reg': hp.quniform('alpha_reg', 0.5, 0.95, 0.05)
            'drop_out': hp.quniform('drop_out', 0.05, 0.30, 0.05)
    }
    num_samples = sum(1 for line in open(args.data))
    loss_weights = get_label_weight_ratio(args.data)
    train_idx, val_idx = train_val_indices(num_samples, 0.8)


    def objective(params, epochs=1500):

        global ITERATION
        ITERATION += 1
        
        params['loss_weights'] = loss_weights
        params['num_classes'] = len(loss_weights)
        params['rnn_hid_size'] = int(params['rnn_hid_size'])
        
        data_train, label_count = data_loader.get_loader(filename=datapath, indices=train_idx, batch_size=batch_size)
        data_val = data_loader.get_loader(filename=datapath, indices=val_idx, batch_size=batch_size)  

        model = myrits.Model()
        print(params)
        model.set_params(**params)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=model.lr)
        early_stopping = EarlyStopping(patience=20, verbose=True, save_mode=1, runname='run_{}'.format(ITERATION), save_path=args.savepath)
        
        start = timer()
        
        val_loss = float('Inf')
        accuracy = 0.0
        for epoch in range(1, epochs+1):
            time_glob = time.time()
            
            train(model, data_train, optimizer, epoch)
            stop_early, val_loss = evaluate(model, data_val, epoch, early_stopping, ITERATION)

            time_ep = time.time() - time_glob
            print('Epoch time {}'.format(time_ep))

            if stop_early:
                break
        run_time = timer() - start
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([val_loss, params, epoch, ITERATION, run_time])
        of_connection.close()
        return {'loss': val_loss, 'params': params, 'iteration': ITERATION, 'train_time': run_time, 'status': STATUS_OK}

 
    bayes_trials = Trials()
    MAX_EVALS=args.max_eval
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=bayes_trials)
    print(best)

if __name__ == '__main__':
    run()
    
