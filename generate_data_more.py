# To run this, load on server:
# module load python_cpu/2.7.14
# module load python_gpu/2.7.14
# module load python_gpu/3.7.1

# module load hdf5/1.10.1
# to run: 
# bsub -n 4 -J gen -W 8:00 -R "rusage[mem=6000,ngpus_excl_p=0]" python generate_data.py --folder patdata_2labels_all

# python3 generate_data.py --folder patdata48_all5_seq10_feat65 --los 48
# python3 generate_data.py --folder patdata24_all5_seq10_feat65 --los 24
# python3 generate_data_more.py --folder patdata24_all8_seq20_feat65_cat8 --los 24 --seq_num 20


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


# add this when each npy file is read in as pandas
def remove_outliers(df2):
    problem_idx = df2.index[df2['vm4'] > 500].tolist() # ABP diastolic
    df2.vm4.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm5'] > 500].tolist() # ABP mean
    df2.vm5.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm6'] > 500].tolist() # NIBPs
    df2.vm6.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm7'] > 500].tolist() # NIBPd
    df2.vm7.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm8'] > 500].tolist() # NIBP mean
    df2.vm8.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm20'] > 100].tolist() # SpO2 level in %. Error if more than 100%
    df2.vm20.iloc[problem_idx] = None    
    problem_idx = df2.index[df2['vm22'] > 200].tolist() # Respiratory rate per minute. normal adult is 12-20 breaths/min
    df2.vm22.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm62'] > 200].tolist() # Ventilator peak pressure, normal 5 to 100 in cmH2O
    df2.vm62.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm131'] > 1000].tolist() # weight in kg
    df2.vm131.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm136'] > 1000].tolist() # lactate
    df2.vm136.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm148'] > 1000].tolist() # K+, filter out one weird value
    df2.vm148.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm149'] > 1000].tolist() # Na+, filter out one weird value
    df2.vm149.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm150'] > 1000].tolist() # Cl-, filter out one weird value
    df2.vm150.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm151'] > 1000].tolist() # Ca2+, filter out one weird value
    df2.vm151.iloc[problem_idx] = None
    problem_idx = df2.index[df2['vm174'] > 10000].tolist() # blood glucose around 100mg/dl, filter out one weird entry
    df2.vm174.iloc[problem_idx] = None
    return df2

# reads the 50 .npy patient data files into pandas danaframe. Reapplies basic data filtering
# to eliminate unreasonable values before returning a list of dataframes
def get_raw_dfs():
    data=[]
    mypath='/cluster/work/grlab/clinical/mimic/MIMIC-III/cdb_1.4/derived_xlyu/filtered_matrix'
    patfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in patfiles:
        newdf = pd.read_hdf(join(mypath, file))
        data.append(remove_outliers(newdf))
    return data
    
# def to_time_bin(x):
#     h, m = map(int, x.split(':'))
#     return h

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


def parse_delta(masks, dir_, los=48):
    num_params = len(masks[0])
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(los):
        if h == 0:
            deltas.append(np.ones(num_params))
        else:
            deltas.append(np.ones(num_params) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, dir_, los=48):
    max_length = los
    deltas = parse_delta(masks, dir_, los)
    
    # only used in GRU-D
    #forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()
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

def parse_id_v2(mimic_id, icu_48, df2, fs, mean, std, topfeats, los=48):
#     print("mimic id: {}".format(mimic_id))
    #DATA_MIN_HOURS = 30 
    # patient record must have 60% non-all zero rows
    DATA_MIN_HOURS = int(0.6*los)

    time0 = icu_48[icu_48['MIMIC_ID'] == mimic_id]['ICU0TIME'].values[0]
    time48 = icu_48[icu_48['MIMIC_ID'] == mimic_id]['ICU48TIME'].values[0]
    filterpatient = df2['MIMIC_ID'] == mimic_id
    filterstart = df2['CHARTTIME'] >= time0
    filterend = df2['CHARTTIME'] < time48
    patient48 = df2[filterpatient & filterstart & filterend]
    patient48['time'] = (patient48['CHARTTIME']-time0).apply(lambda x: to_time_bin(x))
    mycols = ['vm1', 'vm2', 'vm3', 'vm4', 'vm5', 'vm6', 'vm7', 'vm8', 'vm9', 'vm10', 'vm11', 'vm12', 'vm13', 'vm14', 'vm15', 'vm19', 'vm20', 'vm21', 'vm22', 'vm25', 'vm26', 'vm27', 'vm28', 'vm29', 'vm32', 'pm41', 'pm42', 'vm58', 'vm62', 'vm63', 'vm64', 'vm131', 'vm132', 'vm133', 'vm134', 'vm135', 'vm136', 'vm137', 'vm138', 'vm139', 'vm140', 'vm141', 'vm142', 'vm143', 'vm144', 'vm145', 'vm148', 'vm149', 'vm150', 'vm151', 'vm153', 'vm154', 'vm155', 'vm156', 'vm160', 'vm161', 'vm162', 'vm163', 'vm164', 'vm165', 'vm166', 'vm172', 'vm173', 'vm174', 'vm175', 'vm176', 'vm178', 'vm180', 'vm183', 'vm185', 'vm188', 'vm194','time']

    pat_new = pd.DataFrame(columns=mycols)
    patient48 = patient48.drop(['MIMIC_ID', 'SUBJECT_ID','HADM_ID','ICUSTAY_ID','CHARTTIME','WEIGHT'], axis=1)
    nonnan = 0
    
    timelist = patient48['time'].tolist()
    if (len(timelist)==0):
        return 0,0,0
    for h in range(los):
#         print(h)
#         print(patient48['time'].tolist())
        thistime = patient48[patient48['time'] == h]
        num_cols = len(thistime.columns)
        if (len(thistime) > 0):
            # collade these rows with the mean values of each column
            paramvals = thistime.mean().tolist()
            pat_new.loc[len(pat_new)] = paramvals
            nonnan += 1
        else:
            nan_row = [np.nan] * (num_cols - 1)
            pat_new.loc[len(pat_new)] = nan_row + [h]

    if (nonnan < DATA_MIN_HOURS):
        return 0,0,0
    # topfeats = ['vm136','vm1','vm5','vm62','pm41','pm42','vm176','vm13','vm174','vm4','vm20','vm172','vm3']
    evals = pat_new[topfeats]
    evals = evals.values.tolist()
#     og_evals.append(np.array(evals)[:,0])
    og_evals = np.array(evals)
    
    evals = (np.array(evals) - mean) / std
#     norm_evals.append(np.array(evals)[:,0])
    norm_evals = np.array(evals)
    
    shp = evals.shape # (z, 62)
    evals = evals.reshape(-1) # z x 62

    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist() # finds indices where evals is not nan
    notnan = len(indices)
    
    percent = notnan*100/float(len(evals))
#     print("patient {} timesteps: {}, # non-allnan rows {}, # non-nan values: {}".format(mimic_id, nonnan, notnan, percent))

    indices = np.random.choice(indices, len(indices) // 10) # picks 10% of those indices

    values = evals.copy()
    values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    # get ICD code for this entry
    label = icu_48.loc[icu_48['MIMIC_ID'] == mimic_id,'ICD_LABEL'].values[0]
    
    rec = {'label': label}

    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward', los=los)

    rec = json.dumps(rec)

    fs.write(rec + '\n')
    return rec, og_evals, norm_evals

# changes made to adapt for multi class dataset
def get_icu48df(multilabel=1, los=48, seq_num=10):

    los_days = int(los/24)
    #mimic_path = '/cluster/work/grlab/clinical/mimic/MIMIC-III/cdb_1.4'
    #icd9 = pd.read_csv(join(mimic_path, 'derived_xlyu', 'DIAGNOSES_ICD.csv'))
    mimic_path = '/data/chen/thesis/mimic_raw'
    icd9 = pd.read_csv(join(mimic_path, 'DIAGNOSES_ICD.csv'))

    # filter out rows with unrelated codes
    icd9drop = icd9[~icd9['ICD9_CODE'].str.contains('E',na=False)]
    icd9drop = icd9drop[~icd9drop['ICD9_CODE'].str.contains('V',na=False)]
    
    #######################################################################
    # TUNING: TRY WITH DIFFERENT # OF SEQ_NUM -> 7 OR 10
    #######################################################################
    icd9drop = icd9drop[icd9drop['SEQ_NUM'] <= seq_num]
    
    
    # get the main code, first 3 digits
    icd9drop['icd'] = icd9drop['ICD9_CODE'].str[:3]
    icd9drop["icd"] = pd.to_numeric(icd9drop["icd"])
    
    heart = (icd9drop['icd'] >= 390) & (icd9drop['icd'] <= 459) | (icd9drop['icd'] == 785)
    lungs = (icd9drop['icd'] >= 460) & (icd9drop['icd'] <= 519) | (icd9drop['icd'] == 786)
    liver = (icd9drop['icd'] >= 570) & (icd9drop['icd'] <= 579) #liver, pancrease, gallbladder
    gi =  (icd9drop['icd'] >= 530) & (icd9drop['icd'] <= 539) | (icd9drop['icd'] >= 555) & (icd9drop['icd'] <= 569) | (icd9drop['icd'] == 787)
    kidney = (icd9drop['icd'] >= 580) & (icd9drop['icd'] <= 599) | (icd9drop['icd'] == 788) | (icd9drop['icd'] == 791)

    nervous = (icd9drop['icd'] >= 320) & (icd9drop['icd'] <= 359) | (icd9drop['icd'] == 781)
    # endo = (icd9drop['icd'] >= 240) & (icd9drop['icd'] <= 279) | (icd9drop['icd'] == 783)
    endo = (icd9drop['icd'] >= 249) & (icd9drop['icd'] <= 269) | (icd9drop['icd'] == 783)

    blood = (icd9drop['icd'] >= 280) & (icd9drop['icd'] <= 289) | (icd9drop['icd'] == 790)

    types = [heart,lungs,liver,gi,kidney,nervous, endo, blood]
    
    hid = np.sort(icd9drop.HADM_ID.unique())
    d = {'HADM_ID': hid}

    df_pivot = pd.DataFrame(data=d, dtype=np.int64)
    keys = ['heart','lungs','liver','gi','kidney','nervous', 'endo', 'blood']

    for idx, item in enumerate(types):
        df_pivot.loc[:,keys[idx]]=0
        df_pivot.loc[df_pivot.index[df_pivot.HADM_ID.isin(icd9drop[item].HADM_ID.unique())],keys[idx]]=1
        df_pivot.loc[df_pivot.index[df_pivot.HADM_ID.isin(icd9drop[item].HADM_ID.unique())],'type']=keys[idx]

    df_pivot['SUBJECT_ID'] = icd9drop.drop_duplicates(subset=['HADM_ID']).sort_values(by='HADM_ID').SUBJECT_ID.values

    df_pivot = df_pivot[['SUBJECT_ID','HADM_ID','heart','lungs','liver','gi','kidney','nervous', 'endo', 'blood','type']]
    
    ########################################################
    # CHANGE - take samples with only 1 or 2 true labels
    ########################################################
#     count1 = df_pivot.loc[df_pivot.iloc[:,2:].sum(axis=1) == 1]
#     icd_df = df_pivot.loc[(df_pivot.iloc[:,2:].sum(axis=1) == 2) | (df_pivot.iloc[:,2:].sum(axis=1) == 1)]
#     icd_df = df_pivot.loc[(df_pivot.iloc[:,2:].sum(axis=1) == 1)]
    icd_df = df_pivot.loc[(df_pivot.iloc[:,2:7].sum(axis=1) == multilabel)]

    
    ########################################################
    # ICU 48 - take data from the start of ICU + 48 hours
    ########################################################
    #df = pd.read_hdf('/cluster/work/grlab/clinical/mimic/MIMIC-III/cdb_1.4/derived_xlyu/filtered_adm.h5')
    df = pd.read_hdf('/data/chen/thesis/mimic_raw/filtered_adm.h5')
    icu_48 = df[df['ICU_LOS'] >= los]
    dead = icu_48['HOSPITAL_EXPIRE_FLAG']==1
    dpat = icu_48[dead]
    dpat['ICU0TIME'] = dpat['ICUSTARTTIME']
    dpat['ICU48TIME'] = dpat['ICUSTARTTIME'] + pd.Timedelta(days=los_days)
    
    live = icu_48['HOSPITAL_EXPIRE_FLAG']==0
    lpat = icu_48[live]
    lpat['ICU0TIME'] = lpat['ICUSTARTTIME']
    lpat['ICU48TIME'] = lpat['ICUSTARTTIME'] + pd.Timedelta(days=los_days)
    icu_48 = pd.concat([dpat,lpat])
    
    # Find entries that are in icd_48 and icd_df
    patset = list(set(icu_48['HADM_ID']).intersection(set(icd_df['HADM_ID'])))
    patdf = icd_df[icd_df['HADM_ID'].isin(patset)]
    col = patdf['type']
    patdf = patdf.drop(['type'], axis=1)
    patdf['type'] = col
    patdf['ICD_LABEL'] = patdf.iloc[:,2:10].values.tolist()
    # label is now multi-hot encoding
    
    # TODO: Currently ignore duplicated ICU_ID, deal with it later
    icu_48_short = icu_48[icu_48['HADM_ID'].isin(patset)] # make icu_48 match patdf
    icu_48_short = icu_48_short.sort_values(by='HADM_ID')
    # get rid of duplicated HADM_ID
    dup = icu_48_short[icu_48_short.duplicated(subset='HADM_ID')].ICUSTAY_ID.values
    icu_48_short = icu_48_short[~icu_48_short.ICUSTAY_ID.isin(dup)]
    # make patdf match icu_48_short to have same # of rows
    patdf = patdf[patdf['HADM_ID'].isin(icu_48_short.HADM_ID.values)]

    icu_48_short['ICD_LABEL'] = patdf['ICD_LABEL'].values
    icu_48_short['type'] = patdf['type'].values
    icu_48_short = shuffle(icu_48_short)
    return icu_48_short

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate json dataset for MIMIC data'
    )
    parser.add_argument('--folder', type=str, metavar='D', required=True,
                        help='dataset destination')
    parser.add_argument('--los', type=int, default=48)
    parser.add_argument('--seq_num', type=int, default=10)

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

    mypath='/data/chen/thesis/mimic_raw/filtered_matrix'
    patfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#     icu_48_liver = get_icu48df_w_target('liver')
#     icu_48_gi = get_icu48df_w_target('gi')
#     icu_48_kidney = get_icu48df_w_target('kidney')
#     icu_48_kidney = icu_48_kidney[:1000]
    #multilabel=3
    los = args.los
    seq_num = args.seq_num
    icu_48 = get_icu48df(1, los, seq_num)
    icu_48_2 = get_icu48df(2, los, seq_num)
    icu_48_3 = get_icu48df(3, los, seq_num)
    icu_48_4 = get_icu48df(4, los, seq_num)
    icu_48_5 = get_icu48df(5, los, seq_num)
    icu_48_6 = get_icu48df(6, los, seq_num)
    icu_48_7 = get_icu48df(7, los, seq_num)
    icu_48_8 = get_icu48df(8, los, seq_num)
    
    # topfeats = ['vm136','vm1','vm5','vm62','pm41','pm42','vm176','vm13','vm174','vm4','vm20','vm172','vm3']
    topfeats = ['vm1', 'vm2', 'vm3', 'vm4', 'vm5', 'vm6', 'vm7', 'vm8', 'vm9', 'vm10', 'vm11', 'vm13', 'vm14', 'vm15', 'vm20', 'vm22', 'vm29', 'vm32', 'pm41', 'pm42', 'vm58', 'vm62', 'vm63', 'vm64', 'vm131', 'vm132', 'vm133', 'vm134', 'vm135', 'vm136', 'vm137', 'vm138', 'vm139', 'vm140', 'vm141', 'vm142', 'vm143', 'vm144', 'vm145', 'vm148', 'vm149', 'vm150', 'vm151', 'vm153', 'vm154', 'vm155', 'vm156', 'vm160', 'vm161', 'vm162', 'vm163', 'vm164', 'vm165', 'vm166', 'vm172', 'vm173', 'vm174', 'vm175', 'vm176', 'vm178', 'vm180', 'vm183', 'vm185', 'vm188', 'vm194']

    filter_msdf = pd.read_csv("filter-mean-std-30.csv")
    f = filter_msdf[topfeats]
    mean = f.values.tolist()[0]
    std = f.values.tolist()[1]

    for file in patfiles:
        data = pd.read_hdf(join(mypath, file))
        data = remove_outliers(data)
        for mimic_id in data['MIMIC_ID'].unique():
            if mimic_id in icu_48['MIMIC_ID'].tolist():
                rec, og_evals, norm_evals = parse_id_v2(mimic_id, icu_48, data, fs, mean, std, topfeats, los)
            if mimic_id in icu_48_2['MIMIC_ID'].tolist():
                rec, og_evals, norm_evals = parse_id_v2(mimic_id, icu_48_2, data, fs, mean, std, topfeats, los)
            if mimic_id in icu_48_3['MIMIC_ID'].tolist():
                rec, og_evals, norm_evals = parse_id_v2(mimic_id, icu_48_3, data, fs, mean, std, topfeats, los)
            if mimic_id in icu_48_4['MIMIC_ID'].tolist():
                rec, og_evals, norm_evals = parse_id_v2(mimic_id, icu_48_4, data, fs, mean, std, topfeats, los)
            if mimic_id in icu_48_5['MIMIC_ID'].tolist():
                rec, og_evals, norm_evals = parse_id_v2(mimic_id, icu_48_5, data, fs, mean, std, topfeats, los)
            if mimic_id in icu_48_6['MIMIC_ID'].tolist():
                rec, og_evals, norm_evals = parse_id_v2(mimic_id, icu_48_6, data, fs, mean, std, topfeats, los)
            if mimic_id in icu_48_7['MIMIC_ID'].tolist():
                rec, og_evals, norm_evals = parse_id_v2(mimic_id, icu_48_7, data, fs, mean, std, topfeats, los)
            if mimic_id in icu_48_8['MIMIC_ID'].tolist():
                rec, og_evals, norm_evals = parse_id_v2(mimic_id, icu_48_8, data, fs, mean, std, topfeats, los)

#                if rec != 0:
#                    og_ls.append(og_evals)
#                    norm_ls.append(norm_evals)

    #np.save('./{}/og_evals'.format(foldername), og_ls)
    #np.save('./{}/norm_evals'.format(foldername), norm_ls)
