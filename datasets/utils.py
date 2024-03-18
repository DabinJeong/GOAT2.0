import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy

def plot_gene_exp(exp, gene, label):
    exp_with_label = deepcopy(exp)
    exp_with_label.index = list(map(lambda x:x[:-1], exp_with_label.index))
    exp_with_label['patient_label'] = exp_with_label.index.map(lambda x:label[x] if x in list(label.keys()) else None)
    exp_with_label.dropna(inplace=True)
    fig = plt.figure(figsize=(3,3))
    ax = sns.violinplot(exp_with_label, x='patient_label', y=gene, hue='patient_label', legend=False)
    ax.legend().set_visible(False)
    plt.ylabel("TPM")
    plt.title(gene)
    plt.show()
    return

def plot_protein_abundance(exp, gene, label):
    exp_with_label = deepcopy(exp)
    exp_with_label.index = list(map(lambda x:x[:-1], exp_with_label.index))
    exp_with_label['patient_label'] = exp_with_label.index.map(lambda x:label[x] if x in list(label.keys()) else None)
    exp_with_label.dropna(inplace=True)
    fig = plt.figure(figsize=(3,3))
    ax = sns.violinplot(exp_with_label, x='patient_label', y=gene, hue='patient_label', legend=False)
    ax.legend().set_visible(False)
    plt.ylabel("Peptide abundance")
    plt.title(gene)
    plt.show()
    return

def plot_genes_dist(exp, genes, label):
    from scipy.stats import wasserstein_distance
    exp_with_label = deepcopy(exp)
    exp_with_label.index = list(map(lambda x:x[:-1], exp_with_label.index))
    exp_with_label['patient_label'] = exp_with_label.index.map(lambda x:label[x] if x in list(label.keys()) else None)
    groups = exp_with_label['patient_label'].unique()

    li_exp = []
    for group in groups:
        genes_common = list(set(exp.columns).intersection(genes))
        exp_group_tmp = exp_with_label.loc[lambda x:x.patient_label == group,genes_common].median()
        li_exp.append(exp_group_tmp)
    
    plt.hist(li_exp[0], bins=20, label=groups[0])
    plt.hist(li_exp[1], bins=20, label=groups[1])

    w1_dist = wasserstein_distance(li_exp[0], li_exp[1])
    plt.text(x=0.3,y=3,s="W1 distance: {:.3f}".format(w1_dist))
    # plt.xlim(-0.01,0.5)
    plt.legend()
    plt.xlabel("TPM")
    plt.ylabel("# of genes")
    plt.show()
    return

