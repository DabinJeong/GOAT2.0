import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from goat.layers import GraphTransformerLayer

class GOAT_v2(nn.Module):
    def __init__(self, net_params, **kwargs):
        
        super().__init__()

        num_nodes = net_params['num_nodes'] # number of nodes in a graph
        output_dim = net_params['output_dim'] # output dimension of a node
        
        dropout = net_params['dropout']

        patient_dim = output_dim * num_nodes
        
        self.gene_embedding = GOAT_v2_geneEmbedding(net_params)
        self.classifier = classifier(patient_dim//16, 1, dropout)
        
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, g, h, pos_enc):
        h = self.gene_embedding(g, h, pos_enc)
        h = self.classifier(h)
        return h


class GOAT(nn.Module):
    def __init__(self, net_params, **kwargs):
        
        super().__init__()

        num_nodes = net_params['num_nodes'] # number of nodes in a graph
        output_dim = net_params['output_dim'] # output dimension of a node
        
        dropout = net_params['dropout']

        patient_dim = output_dim * num_nodes
        
        self.gene_embedding = GOAT_geneEmbedding(net_params)
        self.classifier = classifier(patient_dim//16, 1, dropout)
        
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, g, h):
        h = self.gene_embedding(g, h)
        h = self.classifier(h)
        return h
    

class MLP(nn.Module):
    def __init__(self, net_params, **kwargs):
        
        super().__init__()

        num_nodes = net_params['num_nodes'] # number of nodes in a graph
        output_dim = net_params['output_dim'] # output dimension of a node
        
        dropout = net_params['dropout']

        patient_dim = output_dim * num_nodes
        
        self.patient_embedding = MLP_embedding(net_params)
        self.classifier = classifier(patient_dim//16, 1, dropout)
        
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, g, h):
        h = self.patient_embedding(g, h)
        h = self.classifier(h)
        return h



class MLP_embedding(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_nodes = net_params['num_nodes'] # number of nodes in a graph
        input_dim = net_params['input_dim'] # input dimension of a node
        
        dropout = net_params['dropout']

        patient_dim = input_dim * num_nodes

        self.layer1 = nn.Linear(patient_dim, patient_dim//16)
        self.batch_norm1 = nn.BatchNorm1d(patient_dim//16)

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self,g,h):
        h = h.float()
        h = h.view(g.batch_size, -1)

        h = self.layer1(h)
        h = F.leaky_relu(self.batch_norm1(h))
        h = self.dropout_layer(h)

        return h
    

class GOAT_v2_geneEmbedding(nn.Module):
    def __init__(self, net_params, **kwargs):
        
        super().__init__()

        num_nodes = net_params['num_nodes'] # number of nodes in a graph
        input_dim = net_params['input_dim'] # input dimension of a node
        hidden_dim = net_params['hidden_dim'] # hidden dimension of a node
        output_dim = net_params['hieen_dim'] # output dimension of a node
        num_heads = net_params['num_heads']
        num_layers = net_params['num_layers']
        assert hidden_dim >= num_heads, "Hidden node dimension in graph attention layer should be greater or equal to the number of heads"
        assert hidden_dim % num_heads == 0, "(Hidden dimension) % (number of heads) should be 0"

        dropout = net_params['dropout']
        residual = net_params['residual']
        batch_norm = net_params['batch_norm']
        layer_norm = net_params['layer_norm']

        patient_dim = output_dim * num_nodes

        self.inp_embedding = nn.Linear(input_dim, hidden_dim)
        self.inp_feat_dropout = nn.Dropout(p=dropout)

        self.pos_encoding = nn.Linear(hidden_dim, hidden_dim)

        self.conv_layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout, residual, batch_norm, layer_norm) for _ in range(num_layers-1)])
        self.conv_layers.append(GraphTransformerLayer(hidden_dim, output_dim, num_heads, dropout, residual, batch_norm, layer_norm))
        
        self.layer1 = nn.Linear(patient_dim, patient_dim//16)
        self.batch_norm1 = nn.BatchNorm1d(patient_dim//16)
        
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, g, h, pos_enc):
        # inp embedding
        h = self.inp_embedding(h)
        h = self.inp_feat_dropout(h)

        # positional encoding embedding
        pos_encoding = self.pos_encoding(pos_enc) 
        h = h + pos_encoding

        # node embedding by GNN
        for conv in self.conv_layers:
            h = conv(g, h)

        # Graph node concat
        h = h.view(g.batch_size, -1)

        # MLP
        h = self.layer1(h)
        h = F.leaky_relu(self.batch_norm1(h))
        h = self.dropout_layer(h)

        return h

class GOAT_geneEmbedding(nn.Module):
    def __init__(self, net_params, **kwargs):
        
        super().__init__()

        num_nodes = net_params['num_nodes'] # number of nodes in a graph
        input_dim = net_params['input_dim'] # input dimension of a node
        hidden_dim = net_params['hidden_dim'] # hidden dimension of a node
        output_dim = net_params['hidden_dim'] # output dimension of a node
        num_heads = net_params['num_heads']
        num_layers = net_params['num_layers']
        assert hidden_dim >= num_heads, "Hidden node dimension in graph attention layer should be greater or equal to the number of heads"
        assert hidden_dim % num_heads == 0, "(Hidden dimension) % (number of heads) should be 0"

        dropout = net_params['dropout']
        residual = net_params['residual']
        batch_norm = net_params['batch_norm']
        layer_norm = net_params['layer_norm']

        patient_dim = output_dim * num_nodes

        self.inp_embedding = nn.Linear(input_dim, hidden_dim)
        self.inp_feat_dropout = nn.Dropout(p=dropout)

        self.conv_layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout, residual, batch_norm, layer_norm) for _ in range(num_layers-1)])
        self.conv_layers.append(GraphTransformerLayer(hidden_dim, output_dim, num_heads, dropout, residual, batch_norm, layer_norm))
        
        self.layer1 = nn.Linear(patient_dim, patient_dim//16)
        self.batch_norm1 = nn.BatchNorm1d(patient_dim//16)
        
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, g, h):
        # inp embedding
        h = self.inp_embedding(h)
        h = self.inp_feat_dropout(h)

        # node embedding by GNN
        for conv in self.conv_layers:
            h = conv(g, h)

        # Graph node concat
        h = h.view(g.batch_size, -1)

        # MLP
        h = self.layer1(h)
        h = F.leaky_relu(self.batch_norm1(h))
        h = self.dropout_layer(h)

        return h

class classifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        
        self.layer1 = nn.Linear(input_dim, input_dim*2)
        self.layer2 = nn.Linear(input_dim*2, input_dim//2)
        self.layer3 = nn.Linear(input_dim//2, output_dim)

        self.batch_norm1 = nn.BatchNorm1d(input_dim*2)
        self.batch_norm2 = nn.BatchNorm1d(input_dim//2)
        self.batch_norm3 = nn.BatchNorm1d(output_dim)

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, h):
        h = self.layer1(h)
        h = F.leaky_relu(self.batch_norm1(h))
        h = self.dropout_layer(h)

        h = self.layer2(h)
        h = F.leaky_relu(self.batch_norm2(h))
        h = self.dropout_layer(h)

        h = self.layer3(h)
        h = self.batch_norm3(h)

        return h