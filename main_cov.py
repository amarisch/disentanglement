# Run this file with:
# module load python_gpu/2.7.14
# module load python_gpu/3.7.1

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
from models import myrits_cov
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
from pytorchtools import EarlyStopping
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from itertools import cycle
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import warnings
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()

# parser.add_argument('--model', type=str, default=myrits)

parser.add_argument('--hyper', type=str, metavar='FILE',
                        help='path of the file of hyperparameters to use ' +
                             'for training; must be a JSON file',
                        default='hparam.json')
parser.add_argument('--data', type=str)
parser.add_argument('--runname', type=str, default='run')
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1000)

# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--hid_size', type=int, default=50)
# parser.add_argument('--impute_weight', type=float, default=1)
# parser.add_argument('--label_weight', type=float, default=2)
# parser.add_argument('--reg', type=int, default=1)
# parser.add_argument('--lambda_reg', type=int, default=4)
# parser.add_argument('--num_classes', type=int, default=5)

parser.add_argument('--seqlen', type=int, default=48)
parser.add_argument('--paramlen', type=int, default=13)
parser.add_argument('--save', type=int, default=1, help='Save the trained model: 1=>save state_dict, 2=>save checkpoint to continue training later')
parser.add_argument('--load', type=int, default=0, help='Load the trained model: 0=>(default) no loading, 1=>load state_dict, 2=>load complete model')
parser.add_argument('--loadpath', type=str)
parser.add_argument('--savepath', type=str)
parser.add_argument('--inference', action='store_true', help='Set this flag to skip training')
parser.add_argument('--equalweights', action='store_true')
parser.add_argument('--noearlystopping', action='store_true')


args = parser.parse_args()

f = open(os.path.join(args.savepath, args.runname + '_out'), 'w')

def plot(recall, precision, auprc, method, epoch, n_classes=5):
    # setup plot details
    # colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='grey',linestyle='--', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(auprc["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color,marker='.', lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, auprc[i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall of {}'.format(method))
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig(os.path.join(args.savepath, "{}_e{}.png".format(method, epoch)))
    # plt.show()

def sklearn_evaluation(y_val, preds, epoch):
    f.write("Epoch {}=\n".format(epoch))
    n_classes = y_val.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    auprc = dict()
    for i in range(n_classes):
        predicted = preds[:, i]
        actual = y_val[:, i]
        precision[i], recall[i], _ = precision_recall_curve(actual, predicted)
        average_precision[i] = average_precision_score(actual, predicted)
        auprc[i] = auc(recall[i], precision[i])
        # f.write('\t\tPrecision:Recall= %.3f:%.3f \n' % (i, precision[i], recall[i]))
        f.write('Class %d ROC PRC= %.3f\n' % (i, auprc[i]))
        f.write('\tAP Score= %.3f\n' % (average_precision[i]))

    # plot curve https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    auroc = roc_auc_score(y_val, preds)
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_val.ravel(), preds.ravel())
    auprc["micro"] = auc(recall["micro"], precision["micro"])
    average_precision["micro"] = average_precision_score(y_val, preds, average="micro")
    f.write('AUPRC over all classes: {0:0.2f}\n'.format(auprc["micro"]))
    #print('AP score over all classes: {0:0.2f}'.format(average_precision["micro"]))
    return recall, precision, auprc   


def get_label_weight_ratio(data):
    content = open('{}'.format(data)).readlines()
    labels = []
    for item in content:
        d = ujson.loads(item)
        labels.append(d['label'])
    labels = np.array(labels)
    labe = np.array([np.where(r==1)[0][0] for r in labels])
    unique, counts = np.unique(labe, return_counts=True)
    bettercounts = np.sum(labels, axis=0)
    
    if args.equalweights:
        return [1.0]* len(unique)
    #weights = []
    #for number in unique:
    #    ratio = round(sum(np.where(unique == number, 0, counts[unique]))/counts[number])
    #    weights.append(ratio)
    #return weights
    return np.round(max(bettercounts)/bettercounts).tolist()

def precision_score_by_class(labels, scores, epoch, method):
    n_classes = 5
    rnn_score_df = pd.DataFrame(scores)
    rnn_label_df = pd.DataFrame(labels)
    rnn_score_arr = []
    rnn_label_arr = []
    for i in range(0,n_classes):
        rnn_score_arr.append(rnn_score_df.loc[:, i].to_numpy())
        rnn_label_arr.append(rnn_label_df.loc[:, i].to_numpy())
    for i in range(0,n_classes):
        ap = average_precision_score(rnn_label_arr[i], rnn_score_arr[i])
        f.write('Average precision-recall score - {} ({}): {}\n'.format(method, i, round(ap, 4)))
        pre, recall, _ = precision_recall_curve(rnn_label_arr[i], rnn_score_arr[i])
        np.save('{}/{}_{}_precision_epoch{}_c{}'.format(args.savepath, args.runname, method, epoch, i), pre)
        np.save('{}/{}_{}_recall_epoch{}_c{}'.format(args.savepath, args.runname, method, epoch, i), recall)
        plt.figure()
        plt.step(recall, pre, where='post')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score: AP={}'.format(round(ap, 4)))
        plt.savefig('{}/{}_{}_prc_epoch{}_c{}'.format(args.savepath, args.runname, method, epoch, i))
        plt.clf()
        plt.cla()

def train(model, data_iter, optimizer, epoch):
    '''
        Trains 1 epoch of the model
    '''
    model.train()

    run_loss = 0.0
    index = 0
    xloss = 0.0
    yloss = 0.0
    for idx, data in enumerate(data_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, optimizer, epoch)
        run_loss += ret['loss'].item()
        xloss += ret['xloss'].item() 
        yloss += ret['yloss'].item() 

    print('\n{} Progress epoch {}, average loss {}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), epoch, run_loss/len(data_iter)))
    f.write('{} Progress epoch {}, average loss {}, {}, {}\n'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),epoch, run_loss/(len(data_iter)*1.0), xloss/(len(data_iter)*1.0),yloss/(len(data_iter)*1.0)))
        
def get_train_conf(model, train_iter, epoch):
    labels = []
    scores = []
    evals = []
    imputations = []
    yloss = 0.0
    save_impute = []
    save_label = []
    save_hidden = []
    save_weight = []
    
    for idx, data in enumerate(train_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        yloss += ret['yloss'].item() 

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        hidden = ret['h'].data.cpu().numpy()
        weight = ret['weight'].data.cpu().numpy()

        save_impute.append(imputation)
        save_label.append(label)
        save_hidden.append(hidden)
        save_weight.append(weight)

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        labels += label.tolist()
        scores += pred.tolist()

    save_impute = np.concatenate(save_impute, axis=0)
    save_label = np.concatenate(save_label, axis=0)
    save_hidden = np.concatenate(save_hidden, axis=0)
    np.save('{}/{}_train_data'.format(args.savepath, args.runname), save_impute)
    np.save('{}/{}_train_label'.format(args.savepath, args.runname), save_label)
    np.save('{}/{}_train_hidden'.format(args.savepath, args.runname), save_hidden)
    np.save('{}/{}_train_weight'.format(args.savepath, args.runname), save_weight)

    labels = np.asarray(labels).astype('int32')
    scores = np.asarray(scores)

    if (epoch % 10 == 0):
        precision_score_by_class(labels, scores, epoch, "train")

    preds = 1*(scores > 0.5)
    f.write('Training accuracy: {}\n'.format(metrics.accuracy_score(labels, preds)))
    
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    
    f.write('Training MAE: {}\n'.format(np.abs(evals - imputations).mean()))
    f.write('Training MRE: {}\n'.format(np.abs(evals - imputations).sum() / np.abs(evals).sum()))
    f.write('Training F1 score: {}\n'.format(f1_score(labels, preds, average='weighted')))
    report = metrics.classification_report(labels, preds)
    file = open('{}/train/{}_epo{}_report'.format(args.savepath, args.runname, epoch), "w")
    file.write(report)
    file.write("\n")
    file.close()
    #save_confusion = metrics.confusion_matrix(labels, preds)
    save_confusion = metrics.multilabel_confusion_matrix(labels, preds)
    np.save('{}/train/{}_epo{}_conf'.format(args.savepath, args.runname,epoch), save_confusion)

def evaluate(model, train_iter, val_iter, epoch, early_stopping):
    model.eval()

    get_train_conf(model, train_iter, epoch)
    
    labels = []
    preds = []
    evals = []
    imputations = []
    save_impute = []
    save_label = []
    save_hidden = []

    val_yloss = 0.0
    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        val_yloss += ret['yloss'].item() 

        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())        
        save_hidden.append(ret['h'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        labels += label.tolist()
        preds += pred.tolist()

    save_impute = np.concatenate(save_impute, axis=0)
    save_label = np.concatenate(save_label, axis=0)
    save_hidden = np.concatenate(save_hidden, axis=0)
    np.save('{}/{}_val_data'.format(args.savepath, args.runname), save_impute)
    np.save('{}/{}_val_label'.format(args.savepath, args.runname), save_label)
    np.save('{}/{}_val_hidden'.format(args.savepath, args.runname), save_hidden)

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    if (epoch % 10 == 0):
        precision_score_by_class(labels, preds, epoch, "val")

    ypreds = 1*(preds > 0.5)

    f.write('Accuracy: {}\n'.format(metrics.accuracy_score(labels, ypreds)))
    
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    
    f.write('MAE: {}\n'.format(np.abs(evals - imputations).mean()))
    f.write('MRE: {}\n'.format(np.abs(evals - imputations).sum() / np.abs(evals).sum()))
    f.write('F1 score: {}\n'.format(f1_score(labels, ypreds, average='weighted')))

    report = metrics.classification_report(labels, ypreds)
    file = open('{}/val/{}_epo{}_report'.format(args.savepath, args.runname, epoch), "w")
    file.write(report)
    file.write("\n")
    file.close()
    #save_confusion = metrics.confusion_matrix(labels, preds)
    save_confusion = metrics.multilabel_confusion_matrix(labels, ypreds)
    np.save('{}/val/{}_epo{}_conf'.format(args.savepath, args.runname,epoch), save_confusion)
    
    if (epoch % 10 == 0):
        recall, precision, auprc = sklearn_evaluation(labels, preds, epoch)
        plot(recall, precision, auprc, "MIMIC-brits", epoch, 5)

    # early stopping based on yloss
    if early_stopping:
        early_stopping(val_yloss/len(val_iter), model)
    
        if early_stopping.early_stop:         
            recall, precision, auprc = sklearn_evaluation(labels, preds, epoch)
            plot(recall, precision, auprc, "MIMIC-brits", epoch, 5)
        return early_stopping.early_stop   
    
    return False
    
def inference(model, val_iter):
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

    preds = preds.argmax(1)
    labels = labels.argmax(1)
    f.write('Accuracy: {}\n'.format(metrics.accuracy_score(labels, preds)))
    
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    
    f.write('MAE: {}\n'.format(np.abs(evals - imputations).mean()))
    f.write('MRE: {}\n'.format(np.abs(evals - imputations).sum() / np.abs(evals).sum()))

    save_impute = np.concatenate(save_impute, axis=0)
    save_label = np.concatenate(save_label, axis=0)
    np.save('{}/{}_inference_data'.format(args.savepath, args.runname), save_impute)
    np.save('{}/{}_inference_label'.format(args.savepath, args.runname), save_label)

    report = metrics.classification_report(labels, preds)
    file = open('{}/{}_report'.format(args.savepath, args.runname), "w")
    file.write(report)
    file.write("\n")
    file.close()
    #save_confusion = metrics.confusion_matrix(labels, preds)
    save_confusion = metrics.multilabel_confusion_matrix(labels, preds)
    np.save('{}/{}_conf'.format(args.savepath, args.runname), save_confusion)
    

# rnn_hid_size, num_classes, impute_weight, label_weight, regularization, lambda_reg, loss_weights
def get_model(hyper_file, loss_weights):
    hf = open(os.path.join(hyper_file), 'r')
    params = json.load(hf)
    hf.close()
    params['drop_out'] = 0.15 
    params['loss_weights'] = loss_weights
    params['num_classes'] = len(loss_weights)
    params['seq_len'] = args.seqlen
    params['param_len'] = args.paramlen
    model = myrits_cov.Model()
    model.set_params(**params)
    return model

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
    np.save('{}/{}_val_idx'.format(args.savepath, args.runname), val_idx)
    train_idx = np.setdiff1d(indices, val_idx)
    data_train = data_loader.get_loader(filename=args.data, indices=train_idx, batch_size=batch_size)
    data_val = data_loader.get_loader(filename=args.data, indices=val_idx, batch_size=batch_size)  
    return data_train, data_val
        
def run():
    print(torch.__version__)
    num_folds = args.cv
    num_samples = sum(1 for line in open(args.data))
    loss_weights = get_label_weight_ratio(args.data)
    #num_samples = sum(1 for line in open(os.path.join(args.data, 'train')))
    #loss_weights = get_label_weight_ratio(os.path.join(args.data, 'val'))
    #print("loss weights based on val set: {}".format(loss_weights))
    f.write('Class dependent loss weight:\n')
    f.write('{}\n'.format(loss_weights))
#     loss_weights = [1,1,1,1,1]
    # TODO: mechanism to load params via json file
    model = get_model(os.path.join(args.savepath, args.hyper), loss_weights)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))
    print('Model hyperram: {}'.format(model.get_params()))
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    defaultepoch = 1

    if args.load == 1:
        model.load_state_dict(torch.load(args.loadpath))

    if args.load == 2:
        checkpoint = torch.load(args.loadpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        defaultepoch = checkpoint['epoch']
        loss = checkpoint['loss']

    if args.inference:
        indices = np.arange(num_samples).tolist()
        data = data_loader.get_loader(filename=args.data, indices=indices, batch_size=model.batch_size)
        inference(model, data)
        return
    
    if (num_folds > 0):
        for train_idx, val_idx in k_folds(n_splits = num_folds, n_samples=num_samples):
            data_train = data_loader.get_loader(filename=args.data, indices=train_idx, batch_size=model.batch_size)
            data_val = data_loader.get_loader(filename=args.data, indices=val_idx, batch_size=model.batch_size)

            for epoch in range(1, args.epochs+1):
                train(model, data_train, optimizer, epoch)
                stop_early = evaluate(model, data_val, epoch)
    else:
        timelist = []
        
        data_train, data_val = train_val_split(num_samples, 0.8, model.batch_size)
        #data_train = data_loader.get_loader(filename=os.path.join(args.data,'train'), indices=np.array([]), batch_size=model.batch_size)
        #data_val = data_loader.get_loader(filename=os.path.join(args.data,'val'), indices=np.array([]), batch_size=model.batch_size)
        early_stopping = None
        if not args.noearlystopping:
            early_stopping = EarlyStopping(patience=20, verbose=True, save_mode=args.save, runname=args.runname, save_path=args.savepath)
        
        for epoch in range(defaultepoch, args.epochs+1):
            time_glob = time.time()
            
            train(model, data_train, optimizer, epoch)
            stop_early = evaluate(model, data_train, data_val, epoch, early_stopping)
            
            time_ep = time.time() - time_glob
            print('Epoch time {}'.format(time_ep))
            timelist.append(time_ep)
            
            if stop_early:
                break
    
    # save model param
    if not args.inference:
        with open(
            os.path.join(
                args.savepath, args.runname + '_hyperparameters.json'
            ), 'w'
        ) as fp:
            json.dump(model.get_params(), fp)
            
    total = sum(timelist)
    print('Average training time: {}, total time: {}'.format(total/len(timelist), total))
            
def createfolder():
    if not os.path.exists(os.path.join(args.savepath, 'train')):
        try:
            os.mkdir(os.path.join(args.savepath, 'train'))
        except OSError:
            print('Train folder creation error')
            exit()
    if not os.path.exists(os.path.join(args.savepath, 'val')):
        try:
            os.mkdir(os.path.join(args.savepath, 'val'))
        except OSError:
            print('Val folder creation error')
            exit()

if __name__ == '__main__':
    createfolder()
    run()

