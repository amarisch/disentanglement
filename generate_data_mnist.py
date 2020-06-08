# To run this, load on server:
# module load python_cpu/2.7.14
# module load python_gpu/2.7.14

# module load hdf5/1.10.1
# to run: 
# bsub -n 4 -J gen -W 4:00 -R "rusage[mem=6000,ngpus_excl_p=1]" python3 generate_data_mnist.py --folder mnist_brits

import pandas as pd
from os import listdir
from os.path import isfile, join
from datetime import datetime
import numpy as np
import json as json
from utils import *
import os
import os.path
from os import path
import argparse
from sklearn.utils import shuffle
from sklearn import preprocessing

def parse_data(x):
    x = x.set_index('Parameter').to_dict()['Value']

    values = []
    for attr in attributes:
#         if x.has_key(attr):
        if attr in x:
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_delta(masks, dir_):
    num_params = len(masks[0])
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(28):
        if h == 0:
            deltas.append(np.ones(num_params))
        else:
            deltas.append(np.ones(num_params) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, dir_):
    max_length = 28
    deltas = parse_delta(masks, dir_)
    
    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values
    rec = {}
    if (len(deltas) >= max_length):
        rec['values'] = np.nan_to_num(values).tolist()[:max_length]
        rec['masks'] = masks.astype('int').tolist()[:max_length]
        # imputation ground-truth
        rec['evals'] = np.nan_to_num(evals).tolist()[:max_length]
        rec['eval_masks'] = eval_masks.astype('int').tolist()[:max_length]
        rec['forwards'] = forwards.tolist()[:max_length]
        rec['deltas'] = deltas.tolist()[:max_length]     
    else:
        pad = [np.zeros(len(deltas[0]))] * (max_length - len(deltas))
        rec['values'] = np.nan_to_num(values).tolist() + np.array(pad).tolist()
        rec['masks'] = masks.astype('int').tolist() + np.array(pad).tolist()
        # imputation ground-truth
        rec['evals'] = np.nan_to_num(evals).tolist() + np.array(pad).tolist()
        rec['eval_masks'] = eval_masks.astype('int').tolist() + np.array(pad).tolist()
        rec['forwards'] = forwards.tolist() + np.array(pad).tolist()
        rec['deltas'] = deltas.tolist() + np.array(pad).tolist()
    return rec

def to_time_bin(x): #time delta version, might need to write a string version(?)
    day = x.days
    sec = x.seconds
    hour = int(sec / 3600) + (day * 24)
    return hour

def parse_id_v2(data, label):
    shp = data.shape # (28, 28)
    evals = data.reshape(-1) # 28 x 28

    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist() # finds indices where evals is not nan
    notnan = len(indices)
    
    percent = notnan*100/float(len(evals))

    indices = np.random.choice(indices, len(indices) // 10) # picks 10% of those indices

    values = evals.copy()
    # will not remove 10%
    #values[indices] = np.nan
    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))
    evals = evals.reshape(shp)
    values = values.reshape(shp)
    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    rec = {'label': label.tolist()}
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec = json.dumps(rec)
    fs.write(rec + '\n')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate json dataset for MIMIC data'
    )
    parser.add_argument('--folder', type=str, metavar='D', required=True,
                        help='dataset destination')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    og_ls = []
    norm_ls = []

    foldername = args.folder
    if not path.exists(foldername):
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
            exit()
        else:
            print ("Successfully created the directory %s " % foldername)

    fs = open('./{}/json'.format(foldername), 'w')

    train = pd.read_csv(os.path.join("mnist", "train.csv"),dtype = np.float32)
    targets_numpy = train.label.values
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.arange(10))
    y_train = lb.transform(targets_numpy)
    features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization
    x_train = features_numpy.reshape(features_numpy.shape[0],28,28)

    for idx in range(0, len(x_train)):
        parse_id_v2(x_train[idx], y_train[idx])

