# GOAT 2.0

## Installation
~~~
conda create --name goat python=3.9
conda activate goat

conda update -n base -c defaults conda
pip install --upgrade pip
~~~


~~~
pip install -r requirements.txt
~~~

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
~~~
python ./demo/test_on_in_distribution_dataset.py -train True -modelConfig configs/models/model_GOAT.yaml -taskConfig configs/tasks/TCGA-LUAD_TMB.yaml -outDir result_test
~~~

