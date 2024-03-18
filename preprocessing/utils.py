from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os

def data_split(dataset, dict_patient_label, test_ratio=0.1, val_ratio=0.1):
    common_samples = list(set(dataset.index).intersection(dict_patient_label.keys()))
    dataset = dataset.loc[common_samples,:]
    labels = list(dataset.index.map(lambda x:dict_patient_label[x]))
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=test_ratio, shuffle=True, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, shuffle=True, stratify=[labels[idx] for idx in train_val_idx])

    return train_idx, val_idx, test_idx

def data_split_from_file(data_split_df, exp_df, crit):
    target_samples = data_split_df.loc[lambda x:x[crit]==True,:].index
    common_idx = list(set(target_samples).intersection(set(exp_df.index.map(lambda x:x[:-1]))))
    sample_filt = exp_df.index.map(lambda x: True if x[:-1] in common_idx else False)
    idx = exp_df.reset_index()[sample_filt]['index']
    return idx

def data_split(dataset, dict_patient_label, test_ratio=0.1, val_ratio=0.1):
    common_samples = list(set(dataset.index).intersection(dict_patient_label.keys()))
    dataset = dataset.loc[common_samples,:]
    labels = list(dataset.index.map(lambda x:dict_patient_label[x]))
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=test_ratio, shuffle=True, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, shuffle=True, stratify=[labels[idx] for idx in train_val_idx])
    return train_idx, val_idx, test_idx

def data_split_from_file(data_split_df, exp_df, crit):
    target_samples = data_split_df.loc[lambda x:x[crit]==True,:].index
    common_idx = list(set(target_samples).intersection(set(exp_df.index.map(lambda x:x[:-1]))))
    sample_filt = exp_df.index.map(lambda x: True if x[:-1] in common_idx else False)
    idx = exp_df.reset_index()[sample_filt]['index']
    return idx

def dataset_feature_scaling(df, train_idx, val_idx, test_idx):
    train_idx_common = list(set(train_idx).intersection(set(df.index)))
    train_df, scaler = df_feature_scaling(df.loc[train_idx_common,:], train=True)

    val_idx_common = list(set(val_idx).intersection(set(df.index)))
    val_df = df_feature_scaling(df.loc[val_idx_common,:], train=False, scaler=scaler)

    test_idx_common = list(set(test_idx).intersection(set(df.index)))
    test_df = df_feature_scaling(df.loc[test_idx_common,:], train=False, scaler=scaler)
    return train_df, val_df, test_df

def df_feature_scaling(df, train=True, scaler=None):
    if train == True:
        scaler = MinMaxScaler(feature_range=(0,1))
        x = df.to_numpy()
        df_scaled = pd.DataFrame(scaler.fit_transform(x), index=df.index, columns=df.columns)
        return df_scaled, scaler
    else:
        x = df.to_numpy()
        x_scaled = scaler.transform(x)
        df_scaled = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
        df_scaled = df_scaled.map(lambda x:1 if x>1 else x)
        df_scaled = df_scaled.map(lambda x:0 if x<0 else x)
        return df_scaled

def df_2_list_pickle(df, dict_patient_label, path):
    if not os.path.exists(path):
        os.makedirs(path)
    li_GNNinput = []
    for idx, row in enumerate(df.index):
        exp_tmp = df.loc[[row],:]
        li_GNNinput.append({'omics':exp_tmp, 'label':dict_patient_label[row]})
    with open(path + "/singleomics.pickle",'wb') as f:
        pickle.dump(li_GNNinput,f)

        
def df_2_list_pickle_multiomics(df_exp, df_prot, dict_patient_label,path):
    if not os.path.exists(path):
        os.makedirs(path)
    li_GNNinput = []
    for idx, row in enumerate(df_exp.index):
        if row in df_exp.index and row in df_prot.index:
            exp_tmp = df_exp.loc[[row],:]
            prot_tmp = df_prot.loc[[row],:]
            multiomics_tmp = pd.concat([exp_tmp, prot_tmp], axis=0)
            multiomics_tmp.fillna(0, inplace=True)
            li_GNNinput.append({'multiomics':multiomics_tmp, 'label':dict_patient_label[row]})

    with open(path + "/multiomics.pickle",'wb') as f:
        pickle.dump(li_GNNinput,f)