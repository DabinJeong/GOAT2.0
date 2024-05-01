from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os

def data_split_from_file(data_split_df, samples_list, crit):
    target_samples = data_split_df.loc[lambda x:x[crit]==True,:].index
    common_idx = list(set(target_samples).intersection(set(map(lambda x:x, samples_list))))
    # sample_filt = list(map(lambda x: True if x in common_idx else False, samples_list))
    return common_idx

def data_split(samples_list, dict_patient_label, test_ratio=0.1, val_ratio=0.1):
    common_samples = list(set(samples_list).intersection(dict_patient_label.keys()))
    dataset = dataset.loc[common_samples,:]
    labels = list(map(lambda x:dict_patient_label[x],samples_list))
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=test_ratio, shuffle=True, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, shuffle=True, stratify=[labels[idx] for idx in train_val_idx])
    return train_idx, val_idx, test_idx

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

        
def df_2_list_pickle_multiomics(df_omics, dict_patient_label,path):
    joint_df = pd.concat(df_omics, axis=0, join='inner')
    if not os.path.exists(path):
        os.makedirs(path)
    li_GNNinput = []
    for idx, row in enumerate(set(joint_df.index)):
        li_omics = []
        for omics in df_omics:
            omics_tmp = omics.loc[[row],:]
            li_omics.append(omics_tmp)
        multiomics_tmp = pd.concat(li_omics, axis=0)
        multiomics_tmp.fillna(0, inplace=True)
        li_GNNinput.append({'multiomics':multiomics_tmp, 'label':dict_patient_label[row]})

    with open(path + "/multiomics.pickle",'wb') as f:
        pickle.dump(li_GNNinput,f)
