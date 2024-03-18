import random
import numpy as np
import torch
import os
from pathlib import Path
from goat.model import GOAT, GOAT_v2, MLP

def fix_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_networks(config):
    net_params = config.model.params
    if config.model.name == "GOAT_v2":
        model = GOAT_v2(net_params)
    if config.model.name == "GOAT":
        model = GOAT(net_params)
    if config.model.name == "MLP":
        model = MLP(net_params) 
    return model

def load_model_from_save(config, path):
    model = load_networks(config)
    # if not path.exists():
    #     return default
    checkpoint = torch.load(path)
    model = torch.load_state_dict(checkpoint['model_state'])
    return model

def load_opt(config, network):
    learning_rate = config.optim.lr
    weight_decay = config.optim.weight_decay
    lr_decay = config.optim.lr_decay
    if config.optim.optimizer == "Adam":
        opt = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if config.optim.optimizer == "SGD":
        opt = torch.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if config.optim.optimizer == "RMSprop":
        opt = torch.optim.RMSprop(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if config.optim.optimizer == "Adagrad":
        opt = torch.optim.Adagrad(network.parameters(), lr=learning_rate, weight_decay=weight_decay, lr_decay=lr_decay)
    if config.optim.optimizer == "Adadelta":
        opt = torch.optim.Adadelta(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if config.optim.optimizer == "AdamW":
        opt = torch.optim.Adadelta(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return opt


def load(config, restore=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_networks(config)
    model.to(device)
    opt = load_opt(config, model)

    if restore is not None and Path(restore).exists():
        checkpoint = torch.load(restore)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    return model,opt
