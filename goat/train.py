import os
import yaml
import numpy as np
from sklearn import metrics
import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ml_collections.config_dict import ConfigDict
from goat.utils import load, fix_seed, seed_worker
from goat.dataset import load_data
from tensorboardX import SummaryWriter


def train(args):
    fix_seed(args.seed)
    config = ConfigDict(yaml.load( open(args.modelConfig,'r'), yaml.FullLoader))
    config_task = ConfigDict(yaml.load( open(args.taskConfig,'r'), yaml.FullLoader))
    config.update(config_task)

    model_name = config.get("model.name")

    print(model_name)

    outDir = args.outDir
    if os.path.exists(outDir) == False:
        os.makedirs(outDir)

    cacheDir = outDir + "/cache"
    logDir = outDir + "/log_{}".format(model_name)
    try:
        if not os.path.exists(logDir):
            os.makedirs(logDir)
    except OSError:
        print('Error: Creating directory {} and {}'.format(logDir))
    try:
        if not os.path.exists(cacheDir):
            os.makedirs(cacheDir)
    except OSError:
        print('Error: Creating directory {} and {}'.format(cacheDir))


    print('Working on {} dataset!'.format(config.data.name))

    torch.cuda.empty_cache()

    multi_omics = config.get("model.multi_omics")
    train_dataset, val_dataset = load_data(config, multi_omics=multi_omics)
    config.model.params.num_nodes = train_dataset.num_nodes

    train_classWeight, val_classWeight = class_weight(train_dataset), class_weight(val_dataset)

    if model_name == "GOAT_v2":
        train_dataset._add_positional_encoding(config.model.params.hidden_dim)
        val_dataset._add_positional_encoding(config.model.params.hidden_dim)
        
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, worker_init_fn=seed_worker, num_workers=20, persistent_workers=True, generator=g, shuffle=True, collate_fn=collate_dgl)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, worker_init_fn=seed_worker, num_workers=20, generator=g, persistent_workers=True, shuffle=False, collate_fn=collate_dgl)
    
    writer = SummaryWriter(log_dir=logDir)
    # ======== Train model ======== 
    if args.train:
        log_file = open(outDir+"/log_{}.txt".format(model_name),'w')
        model, optimizer = load(config)
        # ------ Number of trainable parameters ------
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("# of parameters:", params)
        # --------------------------------------------

        train_loss, val_loss = [], []
        saved_model = None
        min_val_loss = np.inf
        early_stopper = EarlyStopping()

        scheduler = StepLR(optimizer,10)
        
        real_epochs = 0
        for epoch in range(config.training.n_epochs):
            _, optimizer, epoch_train_auprc, epoch_train_auroc, epoch_train_loss= train_epoch(model, optimizer, train_loader, train_classWeight, seed=args.seed)
            epoch_val_loss, epoch_val_auprc, epoch_val_auroc = evaluate_epoch(model, config, val_loader, val_classWeight, seed=args.seed)
            scheduler.step()

            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                saved_model = model

            if early_stopper.should_stop(model, epoch_val_loss):
                print("Early stopping at epoch {}".format(epoch),file=log_file)
                break

            real_epochs += 1

            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)

            print("Epoch {} train loss: {}".format(epoch, epoch_train_loss),file=log_file)
            print("Epoch {} train auprc: {}".format(epoch, epoch_train_auprc),file=log_file)
            print("Epoch {} train auroc: {}".format(epoch, epoch_train_auroc),file=log_file)
            print("Epoch {} validation loss: {}".format(epoch, epoch_val_loss),file=log_file)
            print("Epoch {} validation auprc: {}".format(epoch, epoch_val_auprc),file=log_file)
            print("Epoch {} validation auroc: {}".format(epoch, epoch_val_auroc),file=log_file)


            writer.add_scalar('train/_loss', epoch_train_loss, epoch)
            writer.add_scalar('val/_loss', epoch_val_loss, epoch)
            writer.add_scalar('train/_auprc', epoch_train_auprc, epoch)
            writer.add_scalar('val/_auprc', epoch_val_auprc, epoch)
            writer.add_scalar('train/_auroc', epoch_train_auroc, epoch)
            writer.add_scalar('val/_auroc', epoch_val_auroc, epoch)

        torch.save(state_dict(saved_model, optimizer), cacheDir+"/model_{}.pt".format(model_name))
        writer.close()
        writer.flush()
    else:
        print("Training finished")

    return model

def train_epoch(model, opt, loader, weight=None, seed=42):
    fix_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.train()
    y_true_li, y_pred_li = [], []
    train_loss = 0
    for iter, (batch_graphs, batch_labels) in enumerate(loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].float().to(device)
        batch_labels = batch_labels.float().to(device)

        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].float().to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_pos_enc = None

        opt.zero_grad()
        if batch_pos_enc == None:
            batch_pred = model.forward(batch_graphs, batch_x)
        else:
            batch_pred = model.forward(batch_graphs, batch_x, batch_pos_enc) 

        if weight != None:
            weight = weight.to(device)
            loss =  F.binary_cross_entropy_with_logits(batch_pred, batch_labels, pos_weight=weight)
        else:
            loss =  F.binary_cross_entropy_with_logits(batch_pred, batch_labels)

        loss.requires_grad_(True)
        loss.backward() 

        opt.step()
        train_loss += loss.detach().item() 

        y_true = batch_labels.cpu().detach().flatten().tolist()
        y_pred = batch_pred.cpu().detach().flatten().tolist()

        y_true_li.extend(y_true)
        y_pred_li.extend(y_pred)

    train_loss /= (iter + 1)
    train_auprc, train_auroc = compute_performance(y_true_li, y_pred_li)

    return model, opt, train_auprc, train_auroc, train_loss

def evaluate_epoch(model, config, loader, weight=None, seed=42):
    fix_seed(seed)
    # cacheDir = outdir + "/cache"
    # model, opt = load(config, restore=cacheDir+"/model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        y_true_li, y_pred_li = [], []
        for iter, (batch_graphs, batch_labels) in enumerate(loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].float().to(device)
            batch_labels = batch_labels.float().to(device)
            
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].float().to(device)
                sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            except:
                batch_pos_enc = None

            if batch_pos_enc == None:
                batch_pred = model.forward(batch_graphs, batch_x)
            else:
                batch_pred = model.forward(batch_graphs, batch_x, batch_pos_enc) 

            if weight != None:
                weight = weight.to(device)
                loss =  F.binary_cross_entropy_with_logits(batch_pred, batch_labels, pos_weight=weight)
            else:
                loss =  F.binary_cross_entropy_with_logits(batch_pred, batch_labels)

            test_loss += loss.detach().item()

            y_true = batch_labels.cpu().detach().flatten().tolist()
            y_pred = batch_pred.cpu().detach().flatten().tolist()


            y_true_li.extend(y_true)
            y_pred_li.extend(y_pred)

    test_auprc, test_auroc = compute_performance(y_true_li, y_pred_li)
    test_loss /= (iter + 1)

    return test_loss, test_auprc, test_auroc


class EarlyStopping(object):
    def __init__(self, patience=10):
        self._min_loss = np.inf
        self._patience = patience
        self.__counter = 0
 
    def should_stop(self, model, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return True
        return False
    
    @property
    def counter(self):
        return self.__counter

def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels)).unsqueeze(1)
    batched_graphs = dgl.batch(graphs)
    return batched_graphs, labels

def state_dict(model, opts, **kwargs):
    state = {
        'model_state':model.state_dict(),
        'optimizer_state':opts.state_dict(),
    }
    state.update(kwargs)
    return state

def compute_performance(true_y_li, y_li):
    auroc = metrics.roc_auc_score(true_y_li,y_li)
    auprc = metrics.average_precision_score(true_y_li,y_li)
    return auprc, auroc

def class_weight(dataset):
    _ ,labels = dataset[:]
    pos_samples = sum(labels)
    neg_samples = len(labels) - pos_samples
    class_imbalance = torch.FloatTensor([neg_samples/pos_samples])
    return class_imbalance


