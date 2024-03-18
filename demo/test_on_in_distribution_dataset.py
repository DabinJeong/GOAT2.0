
import argparse
import yaml
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from ml_collections.config_dict import ConfigDict
from goat.train import train, evaluate_epoch, collate_dgl
from goat.dataset import OmicsDataset_singleomics, OmicsDataset_multiomics
from goat.utils import load_model_from_save


def load_data(config, multi_omics=False):
    data_path = config.data.path
    if multi_omics == False:
        test_data = OmicsDataset_singleomics(data_path+"test")
    else:
        test_data = OmicsDataset_multiomics(data_path+"test")
    return test_data

def test(args):
    config = ConfigDict(yaml.load( open(args.modelConfig,'r'), yaml.FullLoader))
    config_task = ConfigDict(yaml.load( open(args.taskConfig,'r'), yaml.FullLoader))
    config.update(config_task)

    model_name = config.get("model.name")

    print(model_name)

    outDir = args.outDir
    if os.path.exists(outDir) == False:
        os.makedirs(outDir)

    torch.cuda.empty_cache()

    multi_omics = config.get("model.multi_omics")
    test_dataset = load_data(config, multi_omics=multi_omics)

    if model_name == "GOAT_v2":
        test_dataset._add_positional_encoding(config.model.params.hidden_dim)

    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, collate_fn=collate_dgl)

    # ======== Train model ======== 
    if args.train:
        saved_model = train(args)
    else:
        cacheDir = args.cacheDir
        assert os.path.exists(cacheDir), "Cache directory does not exist"
        saved_model = load_model_from_save(config, cacheDir+"/model_{}.pt".format(model_name))

    # ======== Test model ======== 
    try:
        test_loss, test_auprc, test_auroc = evaluate_epoch(saved_model, config, test_loader, seed=args.seed)

        print('==================')
        print('Test loss: {}'.format(test_loss))
        print("Test AUPRC: {:.4f}, AUROC: {:.4f}".format(test_auprc, test_auroc))

    except:
        print("Testing bugged")
        import pdb;pdb.set_trace()
    else:
        print("Testing finished")
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed",type=int,default=42)
    parser.add_argument("-train",type=bool)
    parser.add_argument("-modelConfig")
    parser.add_argument("-taskConfig")
    parser.add_argument("-outDir")
    parser.add_argument("-cacheDir")
    args = parser.parse_args()
    test(args)