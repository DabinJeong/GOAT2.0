import pandas as pd
import networkx as nx
import os
from utils import *
from glob import glob
from ml_collections.config_dict import ConfigDict
import yaml
import argparse

# ======== Prepare dataset ======== 
def prepare_dataset(config, multi_omics=False):
    available_omics = {'methylome':None,'transcriptome':None,'proteome':None}

    methylome_files = glob(config.data.path+'methylome_*')
    if len(methylome_files) != 0:
        methylome_file = methylome_files[0]
        if os.path.isfile(methylome_file) == True:
            methylome_df = pd.read_csv(methylome_file,sep='\t',index_col=0)
            available_omics['methylome'] = methylome_df

    exp_files = glob(config.data.path+'transcriptome_*')
    if len(exp_files) != 0:
        exp_file = exp_files[0]
        if os.path.isfile(exp_file) == True:
            exp_df = pd.read_csv(exp_file,sep='\t',index_col=0)
            available_omics['transcriptome'] = exp_df

    prot_files = glob(config.data.path+'proteome_*')
    if len(prot_files) != 0:
        prot_file = prot_files[0]
        if os.path.isfile(prot_file) == True:
            prot_df = pd.read_csv(prot_file,sep='\t',index_col=0)
            available_omics['proteome'] = prot_df
    
    assert sum([True for omics in available_omics.keys() if available_omics[omics] is not None]) != 0, "No omics data found"

    patient_id_file = glob(config.data.patient_label)[0]
    dict_patient_id = pd.read_csv(patient_id_file, sep='\t',header=None,index_col=0).to_dict()[1]

    samples_common = list(set.intersection(*[set(available_omics[omics].index) for omics in available_omics.keys() if available_omics[omics] is not None]))
    for omics in available_omics:
        if available_omics[omics] is not None:
            available_omics[omics] = available_omics[omics].loc[samples_common,:]

    if os.path.isfile(config.data.data_split_file) == True:
        data_split_df = pd.read_csv(config.data.data_split_file, sep='\t',index_col=0)
        train_idx = data_split_from_file(data_split_df, samples_common, 'train')
        val_idx = data_split_from_file(data_split_df, samples_common, 'val')
        test_idx = data_split_from_file(data_split_df, samples_common, 'test')
        print("Train: {}, Val: {}, Test: {}".format(len(train_idx),len(val_idx),len(test_idx)))
    else:
        train_idx, val_idx, test_idx = data_split(samples_common, dict_patient_id)
        print("Train: {}, Val: {}, Test: {}".format(len(train_idx),len(val_idx),len(test_idx)))

    # Feature scaling
    omics_split = {'methylome':None,'transcriptome':None,'proteome':None}
    for omics_type in available_omics:
        if available_omics[omics_type] is not None:
            omics_split[omics_type] = dataset_feature_scaling(available_omics[omics_type], train_idx, val_idx, test_idx)
        else:
            omics_split[omics_type] = None

    list_omics_type = [omics_type for omics_type in available_omics if available_omics[omics_type] is not None]

    if multi_omics == False:
        omics_type = list_omics_type[0]
        train_omics, val_omics, test_omics = omics_split[omics_type]
        df_2_list_pickle(train_omics, dict_patient_id, path=config.data.path+"train")
        df_2_list_pickle(val_omics, dict_patient_id, path=config.data.path+"validation")
        df_2_list_pickle(test_omics, dict_patient_id, path=config.data.path+"test")
    else:
        train_omics, val_omics, test_omics = [], [], []
        for omics_type in list_omics_type:
            train_omics_tmp, val_omics_tmp, test_omics_tmp = omics_split[omics_type]
            train_omics.append(train_omics_tmp)
            val_omics.append(val_omics_tmp)
            test_omics.append(test_omics_tmp)

        df_2_list_pickle_multiomics(train_omics, dict_patient_id, path=config.data.path+"train")
        df_2_list_pickle_multiomics(val_omics, dict_patient_id, path=config.data.path+"validation")
        df_2_list_pickle_multiomics(test_omics, dict_patient_id, path=config.data.path+"test")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-taskConfig")
    args = parser.parse_args()
    config_dataset = ConfigDict(yaml.load(open(args.taskConfig,'r'), yaml.FullLoader))

    if os.path.isdir(config_dataset.data.path) == False:
        os.makedirs(config_dataset.data.path)

    gene_network = pd.read_csv(config_dataset.data.gene_network, sep='\t')
    with open(config_dataset.data.gene_list) as f:
        gene_list = f.read().strip().split('\n')

    gene_network_filt = gene_network.loc[lambda x:np.logical_and(np.isin(x.protein1, gene_list), np.isin(x.protein2, gene_list)), :]
    G = nx.from_pandas_edgelist(gene_network_filt, source='protein1', target='protein2', edge_attr=True)

    G_cc = sorted(nx.connected_components(G), key=len, reverse=True)
    G_lcc = G.subgraph(G_cc[0])

    nx.to_pandas_edgelist(G_lcc).to_csv(config_dataset.data.path+"/STRING_human.tsv",sep='\t',index=False)

    multi_omics = config_dataset.data.omics_data
    for omics in multi_omics:
        exp = pd.read_csv(multi_omics[omics],sep='\t',index_col=0)
        common_genes = list(set(G_lcc.nodes).intersection(exp.columns))
        exp_common = exp.loc[:,common_genes]
        exp_common.to_csv(config_dataset.data.path+"/{}_Genefilt.tsv".format(omics),sep='\t')

    prepare_dataset(config_dataset, multi_omics=True)
    prepare_dataset(config_dataset, multi_omics=False)