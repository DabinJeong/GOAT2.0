# GOAT 2.0
More extensive and easily applicable deep graph attention mdeol for multi-omics biomarker discovery.
Original manuscript (2023, Bioinformatics) available [**here**](https://academic.oup.com/bioinformatics/article/39/10/btad582/7280697)

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
~~~
python preprocessing/preprocessing.py -taskConfig ./configs/tasks/TCGA-LUAD_TMB.yaml 
~~~

## Experiments
~~~
python ./demo/test_on_in_distribution_dataset.py -train True -modelConfig configs/models/model_GOAT.yaml -taskConfig configs/tasks/TCGA-LUAD_TMB.yaml -outDir result_test
~~~

