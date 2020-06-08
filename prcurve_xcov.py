import numpy as np
import pandas as pd
import os
import os.path
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score
import argparse
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

from models import myrits_xcov
import os
import sys

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import ujson as ujson
import json
import data_loader
from itertools import cycle
import utils
import time
import warnings
warnings.simplefilter("ignore")

parser= argparse.ArgumentParser()
parser.add_argument('--t', type=str, default='0.5', help='feature filtering threshold')
parser.add_argument('--loadpath', type=str)
parser.add_argument('--runname', type=str, default='run')
parser.add_argument('--data', type=str)
parser.add_argument('--seqlen', type=int, default=48)

args = parser.parse_args()


t_list = [float(item) for item in args.t.split(',')]

f = open(os.path.join(args.loadpath, args.runname + '_prc_out.csv'), 'w')

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
    return np.round(max(bettercounts)/bettercounts).tolist()

def get_model(loss_weights):
    hf = open(os.path.join(args.loadpath, args.runname + "_hyperparameters.json"), 'r')
    params = json.load(hf)
    hf.close()
    params['drop_out'] = 0.15 
    params['loss_weights'] = loss_weights
    params['num_classes'] = len(loss_weights)
    params['seq_len'] = args.seqlen
    model = myrits_xcov.Model()
    model.set_params(**params)
    return model

def inference(model, val_iter):
    model.eval()

    labels = []
    preds = []
    evals = []
    imputations = []
    save_impute = []
    save_label = []
    save_hidden = []
    save_weight = []
    
    val_yloss = 0.0
    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        val_yloss += ret['yloss'].item() 

        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())
        save_hidden.append(ret['h'].data.cpu().numpy())
        save_weight.append(ret['weight'].data.cpu().numpy())
        
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
    save_hidden = np.concatenate(save_hidden, axis=0)

    # evals = np.asarray(evals)
    # imputations = np.asarray(imputations)
    
    # f.write('MAE: {}\n'.format(np.abs(evals - imputations).mean()))
    # f.write('MRE: {}\n'.format(np.abs(evals - imputations).sum() / np.abs(evals).sum()))

    # save_impute = np.concatenate(save_impute, axis=0)
    # save_label = np.concatenate(save_label, axis=0)
    # np.save('{}/{}_inference_data'.format(args.savepath, args.runname), save_impute)
    # np.save('{}/{}_inference_label'.format(args.savepath, args.runname), save_label)

    # report = metrics.classification_report(labels, preds)
    # file = open('{}/{}_report'.format(args.savepath, args.runname), "w")
    # file.write(report)
    # file.write("\n")
    # file.close()
    #save_confusion = metrics.confusion_matrix(labels, preds)
    # save_confusion = metrics.multilabel_confusion_matrix(labels, preds)
    # np.save('{}/{}_conf'.format(args.savepath, args.runname), save_confusion)
    return preds, labels, save_hidden, save_weight

def np_to_pd(matrix):
    if matrix.shape[0] == 5:
        df = pd.DataFrame(data=matrix.transpose(), index=np.arange(matrix.shape[-1]).tolist(), columns=["heart", "lungs", "liver", "gi", "kidney"])
    elif matrix.shape[0] == 8:
        df = pd.DataFrame(data=matrix.transpose(), index=np.arange(matrix.shape[-1]).tolist(), columns=["heart", "lungs", "liver", "gi", "kidney","nervous", "endo", "blood"])        
    else:
        print("Unseen weight matrix dimension: {}".format(matrix.shape))
        exit()
    return df

def plot_prc(classifier, X_test, y_test, category="2-class"):
	# y_score= Target scores, can either be probability estimates of the positive class, confidence values, 
	# or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
	y_score = classifier.decision_function(X_test)
	average_precision = average_precision_score(y_test, y_score)

	print('Average precision-recall score: {0:0.2f}'.format(
	      average_precision))

	disp = plot_precision_recall_curve(classifier, X_test, y_test)
	disp.ax_.set_title('{} Precision-Recall curve: AP={}'.format(category, average_precision))
	plt.savefig(os.path.join(args.loadpath, args.runname + "prc_plot_{}".format(category)))

# def plot_prc(classifier, X_test, y_test, rnn_score, rnn_y, category="2-class"):
# 	# y_score= Target scores, can either be probability estimates of the positive class, confidence values, 
# 	# or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
# 	y_score = classifier.decision_function(X_test)
# 	average_precision = average_precision_score(y_test, y_score)
# 	rnn_ap = average_precision_score(rnn_y, rnn_score)
# 	print('Average precision-recall score, RNN - LR: {}   {}'.format(
# 	      round(rnn_ap, 4), round(average_precision, 4)))

# 	# colors = cycle(['navy', 'darkorange'])
# 	pre, recall, _ = precision_recall_curve(y_test, y_score)
# 	rnn_pre, rnn_recall, _ = precision_recall_curve(rnn_y, rnn_score)

# 	f1_score = f1_score(y_test, classifier.predict(X_test))
# 	rnn_pred = 1* (rnn_y >= 0.5)
# 	rnn_fi_score = f1_score(rnn_y, rnn_pred)

# 	# disp = plot_precision_recall_curve(classifier, X_test, y_test)
# 	# disp.ax_.set_title('{} Precision-Recall curve: AP={}'.format(category, average_precision))
	
# 	lines = []
# 	labels = []
# 	l, = plt.plot(recall, pre, color='darkorange', lw=2)
# 	lines.append(l)
# 	labels.append('Precision-recall for LR (area = {})'.format(average_precision))
# 	l, = plt.plot(rnn_recall, rnn_pre, color='blue', lw=2)
# 	lines.append(l)
# 	labels.append('Precision-recall for RNN (area = {})'.format(rnn_ap))
# 	fig = plt.gcf()
# 	fig.subplots_adjust(bottom=0.25)
# 	plt.xlim([0.0, 1.0])
# 	plt.ylim([0.0, 1.05])
# 	plt.xlabel('Recall')
# 	plt.ylabel('Precision')
# 	plt.title('Extension of Precision-Recall curve to multi-class')
# 	plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

# 	plt.savefig(os.path.join(args.loadpath, args.runname + "_prc_plot_t{}_{}".format(str(args.t).replace('.',''), category)))
# 	plt.cla()
# 	plt.clf()

from sklearn.metrics import precision_recall_fscore_support

def compute_prc(classifier, X_test, y_test, X_train, y_train, rnn_score_test, rnn_y_test, rnn_score_train, rnn_y_train, category, threshold):
    # y_score= Target scores, can either be probability estimates of the positive class, confidence values, 
    # or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
    y_score_train = classifier.decision_function(X_train)
    ap_train = average_precision_score(y_train, y_score_train)
    rnn_ap_train = average_precision_score(rnn_y_train, rnn_score_train)
    print('Train average precision score, RNN - LR: {}   {}'.format(
          round(rnn_ap_train, 4), round(ap_train, 4)))

    y_score = classifier.decision_function(X_test)
    ap = average_precision_score(y_test, y_score)
    rnn_ap = average_precision_score(rnn_y_test, rnn_score_test)
    print('Val average precision score, RNN - LR: {}   {}'.format(
          round(rnn_ap, 4), round(ap, 4)))

    y_pred_train = classifier.predict_proba(X_train)[:, 1]
    y_pred_train = 1* (y_pred_train >= 0.5)
    f1score_train = f1_score(y_train, y_pred_train)
    rnn_pred_train = 1* (rnn_score_train >= 0.5)
    rnn_fiscore_train = f1_score(rnn_y_train, rnn_pred_train)
    print('Train average f1 score, RNN - LR: {}   {}'.format(
          round(rnn_fiscore_train, 4), round(f1score_train, 4)))

    # y_pred = classifier.predict(X_test)
    y_pred = classifier.predict_proba(X_test)[:, 1]
    y_pred = 1* (y_pred >= 0.5)
    f1score = f1_score(y_test, y_pred)
    rnn_pred = 1* (rnn_score_test >= 0.5)
    rnn_fiscore = f1_score(rnn_y_test, rnn_pred)

    print('Val average f1 score, RNN - LR: {}   {}'.format(
          round(rnn_fiscore, 4), round(f1score, 4)))
    # print("RNN: {}".format(precision_recall_fscore_support(rnn_y_test, rnn_pred)))
    # print("LR: {}".format(precision_recall_fscore_support(y_test, y_pred)))

    # print("RNN count labels: {} {}".format(np.count_nonzero(rnn_pred==0), np.count_nonzero(rnn_pred==1)))
    # print("LR count labels: {} {}".format(np.count_nonzero(y_pred==0), np.count_nonzero(y_pred==1)))

    pre_train, recall_train, _ = precision_recall_curve(y_train, y_score_train)
    rnn_pre_train, rnn_recall_train, _ = precision_recall_curve(rnn_y_train, rnn_score_train)

    pre, recall, _ = precision_recall_curve(y_test, y_score)
    rnn_pre, rnn_recall, _ = precision_recall_curve(rnn_y_test, rnn_score_test)
    lines = []
    labels = []
    
    l, = plt.plot(recall_train, pre_train, color='darkorange',linestyle='dashed', lw=2)
    lines.append(l)
    labels.append('Train precision-recall for LR (area = {})'.format(round(ap_train, 4)))
    l, = plt.plot(rnn_recall_train, rnn_pre_train, color='blue', linestyle='dashed',lw=2)
    lines.append(l)
    labels.append('Train precision-recall for RNN (area = {})'.format(round(rnn_ap_train, 4)))

    l, = plt.plot(recall, pre, color='darkorange', lw=2)
    lines.append(l)
    labels.append('Val precision-recall for LR (area = {})'.format(round(ap, 4)))
    l, = plt.plot(rnn_recall, rnn_pre, color='blue', lw=2)
    lines.append(l)
    labels.append('Val precision-recall for RNN (area = {})'.format(round(rnn_ap, 4)))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve - t={} - {}'.format(threshold, category))
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    figname = "{}_prc_plot_t{}_{}".format(args.runname, str(threshold).replace('.',''), category)
    plt.savefig(os.path.join(args.loadpath, os.path.join("prc", figname)))
    plt.cla()
    plt.clf()

    return round(ap,4), round(rnn_ap,4), round(f1score,4), round(rnn_fiscore,4)

# def generate_representation(threshold):
#     m1 = np.load(os.path.join(args.loadpath, args.runname + "_train_weight.npy"))
#     d1 = np_to_pd(m1[1])
#     dd1 = d1[(d1>threshold) | (d1 < (-1*threshold))].notnull()

#     types = ['heart','lungs','liver','gi','kidney']
#     t_idx = []
#     feat_len = []
#     for t in types:
#         indices = dd1[dd1[t]==True].index.tolist()
#         t_idx.append(indices)
#         feat_len.append(len(indices))
#         print(t, " ", len(indices), ": ", indices)

#     labels = np.load(os.path.join(args.loadpath, args.runname + '_train_label.npy'))
#     hidden = np.load(os.path.join(args.loadpath, args.runname + '_train_hidden.npy'))

#     vlabels = np.load(os.path.join(args.loadpath, args.runname + '_val_label.npy'))
#     vhidden = np.load(os.path.join(args.loadpath, args.runname + '_val_hidden.npy'))

#     labels = np.concatenate((labels, vlabels), axis=0)
#     hidden = np.concatenate((hidden, vhidden), axis=0)

#     df = pd.DataFrame(hidden)
#     labels_df = pd.DataFrame(labels)

#     data_arr = []
#     label_arr = []
#     for i in range(0,len(types)):
#         data_arr.append(df.loc[ : , t_idx[i]].to_numpy())
#         label_arr.append(labels_df.loc[:, i].to_numpy())
#     return feat_len, data_arr, label_arr

def generate_representation(threshold, hidden, labels, weight, val_idx, train_idx):
    # m1 = np.load(os.path.join(args.loadpath, args.runname + "_train_weight.npy"))
    # d1 = np_to_pd(m1[1])
    d1 = np_to_pd(weight)
    dd1 = d1[(d1>threshold) | (d1 < (-1*threshold))].notnull()

    types = ['heart','lungs','liver','gi','kidney']
    if (weight.shape[0] == 8):
        types = ['heart','lungs','liver','gi','kidney', 'nervous', 'endo', 'blood']
    t_idx = []
    feat_len = []
    for t in types:
        indices = dd1[dd1[t]==True].index.tolist()
        t_idx.append(indices)
        feat_len.append(len(indices))
        print(t, " ", len(indices), ": ", indices)

    val_data = np.array(hidden)[val_idx]
    train_data = np.array(hidden)[train_idx]
    val_label = np.array(labels)[val_idx]
    train_label = np.array(labels)[train_idx]

    val_df = pd.DataFrame(val_data)
    train_df = pd.DataFrame(train_data)
    val_labels_df = pd.DataFrame(val_label)
    train_labels_df = pd.DataFrame(train_label)

    data_arr = []
    label_arr = []
    val_data_arr = []
    val_label_arr = []
    for i in range(0,len(types)):
        data_arr.append(train_df.loc[ : , t_idx[i]].to_numpy())
        label_arr.append(train_labels_df.loc[:, i].to_numpy())
        val_data_arr.append(val_df.loc[ : , t_idx[i]].to_numpy())
        val_label_arr.append(val_labels_df.loc[:, i].to_numpy())
    return feat_len, data_arr, label_arr, val_data_arr, val_label_arr



###############################################################
#       get RNN model for comparison
###############################################################

num_samples = sum(1 for line in open(args.data))
loss_weights = get_label_weight_ratio(args.data)
model = get_model(loss_weights)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=model.lr)
model.load_state_dict(torch.load(os.path.join(args.loadpath, args.runname + "_checkpoint.pt")))
indices = np.arange(num_samples).tolist()
data = data_loader.get_loader(filename=args.data, indices=indices, batch_size=model.batch_size)
preds, labels, hidden, weight = inference(model, data)

types = ['heart','lungs','liver','gi','kidney']
if (model.num_classes == 8):
        types = ['heart','lungs','liver','gi','kidney', 'nervous', 'endo', 'blood']

rnn_score_df = pd.DataFrame(preds)
rnn_label_df = pd.DataFrame(labels)
rnn_score_arr = []
rnn_label_arr = []
for i in range(0,len(types)):
    rnn_score_arr.append(rnn_score_df.loc[:, i].to_numpy())
    rnn_label_arr.append(rnn_label_df.loc[:, i].to_numpy())


# pick out validation indices
indices = np.arange(num_samples)
# val_idx = np.random.choice(indices, num_samples // 5)
val_idx = np.load(os.path.join(args.loadpath, args.runname + "_val_idx.npy"), 'r')
train_idx = np.setdiff1d(indices, val_idx)

if (model.num_classes == 5):
    f.write('t, len, lr_heart, lung, liver, gi, kidney, rnn_heart, lung, liver, gi, kidney\n')
if (model.num_classes == 8):
    f.write('t, len, lr_heart, lung, liver, gi, kidney, nervous, endo, blood, rnn_heart, lung, liver, gi, kidney, nervous, endo, blood\n')

###############################################################
#      get Hidden Representation for LR classifier
###############################################################
for t in t_list:
    print("Threshold: {}".format(t))
    feat_len, train_data, train_label, val_data, val_label = generate_representation(t, hidden, labels, weight[-1], val_idx, train_idx)

    ###############################################################
    #           plot PR curve
    ###############################################################    
    # for i, category in enumerate(types):
    #   clf = LogisticRegression(random_state=0).fit(data_arr[i], label_arr[i])
    #   plot_prc(clf, data_arr[i], label_arr[i], rnn_score_arr[i], rnn_label_arr[i], category)

    ###############################################################
    #           write threshold data to file
    ###############################################################
    lr_ap = []
    lr_f1 = []
    rnn_ap = []
    rnn_f1 = []
    for i, category in enumerate(types):
        clf = LogisticRegression(random_state=0).fit(train_data[i], train_label[i])
        ap, rnn_pre, f1score, rnn_fiscore = compute_prc(clf, val_data[i], val_label[i], \
                                                        train_data[i], train_label[i], \
                                                        np.array(rnn_score_arr[i])[val_idx], np.array(rnn_label_arr[i])[val_idx], \
                                                        np.array(rnn_score_arr[i])[train_idx], np.array(rnn_label_arr[i])[train_idx], \
                                                        category, t)
        lr_ap.append(ap)
        lr_f1.append(f1score)
        rnn_ap.append(rnn_pre)
        rnn_f1.append(rnn_fiscore)

    if not 0 in feat_len:
        avg = sum(feat_len)/len(types)
        # output = "{},{},{},{},{},{}\n".format(t, avg, ','.join([str(x) for x in lr_ap]), ','.join([str(x) for x in lr_f1]),\
        #     ','.join([str(x) for x in rnn_ap]), ','.join([str(x) for x in rnn_f1]))
        output = "{},{},{},{}\n".format(t, avg, ','.join([str(x) for x in lr_ap]),','.join([str(x) for x in rnn_ap]))

        f.write(output)

