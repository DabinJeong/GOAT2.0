import pandas as pd
import networkx as nx
from utils import *
from glob import glob
from ml_collections.config_dict import ConfigDict
import yaml
import argparse

# ======== Prepare dataset ======== 
def prepare_dataset(config, multi_omics=False):
    exp_file = glob(config.data.path+'transcriptome_*')[0]
    exp_df = pd.read_csv(exp_file,sep='\t',index_col=0)

    prot_file = glob(config.data.path+'proteome_*')[0]
    prot_df = pd.read_csv(prot_file,sep='\t',index_col=0)

    patient_id_file = glob(config.data.path+'Patient_label_*')[0]
    dict_patient_id = pd.read_csv(patient_id_file, sep='\t',header=None,index_col=0).to_dict()[1]

    if os.path.isfile(config.data.data_split_file) == True:
        data_split_df = pd.read_csv(config.data.data_split_file, sep='\t',index_col=0)
        train_idx = data_split_from_file(data_split_df, exp_df, 'train')
        val_idx = data_split_from_file(data_split_df, exp_df, 'val')
        test_idx = data_split_from_file(data_split_df, exp_df, 'test')
        print("Train: {}, Val: {}, Test: {}".format(len(train_idx),len(val_idx),len(test_idx)))
    else:
        train_idx, val_idx, test_idx = data_split(exp_df, dict_patient_id)
        print("Train: {}, Val: {}, Test: {}".format(len(train_idx),len(val_idx),len(test_idx)))

    # Feature scaling
    train_exp, val_exp, test_exp = dataset_feature_scaling(exp_df, train_idx, val_idx, test_idx)
    train_prot, val_prot, test_prot = dataset_feature_scaling(prot_df, train_idx, val_idx, test_idx)

    if multi_omics == False:
        df_2_list_pickle(train_exp, dict_patient_id, path=config.data.path+"train")
        df_2_list_pickle(val_exp, dict_patient_id, path=config.data.path+"validation")
        df_2_list_pickle(test_exp, dict_patient_id, path=config.data.path+"test")
    else:
        df_2_list_pickle_multiomics(train_exp, train_prot, dict_patient_id, path=config.data.path+"train")
        df_2_list_pickle_multiomics(val_exp, val_prot, dict_patient_id, path=config.data.path+"validation")
        df_2_list_pickle_multiomics(test_exp, test_prot, dict_patient_id, path=config.data.path+"test")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-taskConfig")
    args = parser.parse_args()
    config_dataset = ConfigDict(yaml.load(open(args.taskConfig,'r'), yaml.FullLoader))

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