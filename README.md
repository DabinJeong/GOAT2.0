# GOAT 2.0
More extensive and easily applicable deep graph attention model for multi-omics biomarker discovery.
Original manuscript available [**here**](https://academic.oup.com/bioinformatics/article/39/10/btad582/7280697).
Here are the major updates in GOAT version 2.0
- Random walk positional encoding of genes in gene-gene interaction graph is added to reflect global structure of the graph (`GOAT_v2` model in `goat/model.py`).
- Base library for GNN implementation transformed from PyG to dgl (https://www.dgl.ai/).
  
## Installation
Create conda environment.
~~~
conda create --name goat python=3.9
conda activate goat

conda update -n base -c defaults conda
pip install --upgrade pip
~~~

Install required packages.
~~~
pip install -r requirements.txt
~~~

Install GOAT2.0.
~~~
pip install -e .
~~~

## Data preprocessing
Gene-gene interaction network from STRING database (https://string-db.org) and gene list to filter the network is required.
In data directory specified in configuration file, omics data (patient X gene) and patient label (patient X label) should be stored.
The following script will generate pickle file to be used to generate custom dataset object in './goat/dataset.py' that inherits torch.Dataset object.
~~~
python preprocessing/preprocessing.py -taskConfig ./configs/tasks/TCGA-LUAD_TMB.yaml 
~~~

## Experiments
You can specify model hyper-parameters in `configs/models/model_*.yaml`. Available models are `MLP`, `GOAT`, `GOAT_v2`.
You can specify datasets and datasplits in `configs/tasks/*.yaml`.
~~~
python ./demo/test_on_in_distribution_dataset.py -train True -modelConfig configs/models/model_GOAT.yaml -taskConfig configs/tasks/TCGA-LUAD_TMB.yaml -outDir result_test
~~~

