import pandas as pd
import numpy as np
from scipy import sparse
import dgl
import torch
from torch.utils.data import Dataset
from dgl import RandomWalkPE
import pickle
from glob import glob
from goat.utils import *

def load_data(config, multi_omics=False):
    data_path = config.data.path
    if multi_omics == False:
        train_data = OmicsDataset_singleomics(data_path+"train")
        val_data = OmicsDataset_singleomics(data_path+"validation")
    else:
        train_data = OmicsDataset_multiomics(data_path+"train")
        val_data = OmicsDataset_multiomics(data_path+"validation")
    return train_data, val_data

class patientDGL_multiomics(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        omics_file = glob(data_dir+'/multiomics.pickle')
        with open(omics_file[0], 'rb') as f:
            self.data = pickle.load(f)

        """
        data is a list of pateint obejcts
        patient is a dict object of gene-level omics measurements in pandas.DataFrame object with following attributes
        
        patient = data[idx]
        patient['multiomics']: (n X k pd.DataFrame) k omics profile of n patients , where is n is the number of genes of interest
        patient['label']: (str) categorical variable that indicates patient label
        """
        topology_file = glob(data_dir+'/../STRING_*.tsv')
        self.topology = pd.read_csv(topology_file[0],sep='\t')
        """
        topology is a pandas.DataFrame object of gene-gene interaction edgelist
        """

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
    
    def _prepare(self):
        for omics in self.data:

            node_features = torch.tensor(omics['multiomics'].values, dtype=torch.float64)
            node_features = node_features.permute(1,0)

            num_nodes = len(node_features)
            g_nodeID = dict(zip(omics['multiomics'].columns, range(num_nodes)))

            edges = self.topology.iloc[:,[0,1]].map(lambda x:g_nodeID[x] if x in g_nodeID else None).dropna().to_numpy()
            srcs, tgts = torch.tensor(edges[:,0]), torch.tensor(edges[:,1])
            g = dgl.graph((srcs, tgts), num_nodes=num_nodes, idtype=torch.int64)

            g.ndata['feat'] = node_features

            self.graph_lists.append(g)
            label = omics['label']
            self.graph_labels.append(label)

        self.num_nodes = num_nodes
 
    def __len__(self):
        return self.n_samples

    def __getitem__(self,idx):
        return self.graph_lists[idx], self.graph_labels[idx]

class patientDGL_singleomics(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        exp_file = glob(data_dir+'/singleomics.pickle')

        with open(exp_file[0], 'rb') as f:
            data_tmp = pickle.load(f)

        patient_ids = [data['omics'].index[0] for data in data_tmp]
        patient_id_idx = pd.DataFrame(enumerate(patient_ids),columns=['idx','patient_id']).sort_values(by='patient_id')['idx'].to_numpy()
        self.data = [data_tmp[i] for i in patient_id_idx]
        """
        data is a list of pateint obejcts
        patient is a dict object of gene-level omics measurements in pandas.DataFrame object with following attributes
        
        patient = data[idx]
        patient['omics']: (n X 1 pd.DataFrame) omics profile of a patient, where is n is the number of genes of interest
        patient['label']: (str) categorical variable that indicates patient label
        """
        topology_file = glob(data_dir+'/../STRING_*.tsv')
        self.topology = pd.read_csv(topology_file[0],sep='\t')
        """
        topology is a pandas.DataFrame object of gene-gene interaction edgelist
        """

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
        
    
    def _prepare(self):
        for omics in self.data:
            node_features = torch.reshape(torch.tensor(omics['omics'].values, dtype=torch.float64), (-1,1))
            num_nodes = len(node_features)
            g_nodeID = dict(zip(omics['omics'].columns, range(num_nodes)))

            edges = self.topology.iloc[:,[0,1]].map(lambda x:g_nodeID[x] if x in g_nodeID else None).dropna().to_numpy()
            srcs, tgts = torch.tensor(edges[:,0]), torch.tensor(edges[:,1])
            g = dgl.graph((srcs, tgts), num_nodes=num_nodes, idtype=torch.int64)

            g.ndata['feat'] = node_features

            self.graph_lists.append(g)
            label = omics['label']
            self.graph_labels.append(label)
        
        self.num_nodes = num_nodes

    def __len__(self):
        return self.n_samples

    def __getitem__(self,idx):
        return self.graph_lists[idx], self.graph_labels[idx]


class OmicsDataset_singleomics(Dataset):
    def __init__(self, data_dir):
        self.dataset = patientDGL_singleomics(data_dir)
        self.num_nodes = self.dataset.num_nodes

    def _add_positional_encoding(self, pos_enc_dim):
        transform = RandomWalkPE(k=pos_enc_dim, feat_name='pos_enc')
        self.graph_lists = [transform(g) for g in self.dataset.graph_lists]
   
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        return self.dataset[idx]
    


class OmicsDataset_multiomics(Dataset):
    def __init__(self, data_dir):
        self.dataset = patientDGL_multiomics(data_dir)
        self.dataset.num_nodes = self.dataset.num_nodes

    def _add_positional_encoding(self, pos_enc_dim):
        transform = RandomWalkPE(k=pos_enc_dim, feat_name='pos_enc')
        self.graph_lists = [transform(g) for g in self.dataset.graph_lists]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        return self.dataset[idx]
    

def laplacian_positional_encoding(g, pos_enc_dim):
    # Laplacian
    A = g.adj_external(scipy_fmt='coo')
    N = sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sparse.eye(g.number_of_nodes()) - N * A * N # <- Symetrically normalized Laplacian

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g
