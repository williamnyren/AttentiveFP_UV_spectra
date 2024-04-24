import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
from decimal import Decimal

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from AttentiveFP_GATv2conv import AttentiveFP
from torch_geometric.loader import DataListLoader, DataLoader

from torch_geometric.data import Data, OnDiskDataset, download_url, extract_zip
from torch_geometric.data.data import BaseData, Data
from torch_geometric_modified import from_smiles

from data_utils import GenSplit, DatasetAttentiveFP
from sweep_utils import find_batch_size

import params
import time
import yaml

from tqdm import tqdm
import wandb

WORKSPACE_PATH = '/home/nyrenw/Documents/workspace/AttentiveFP'
MAX_NUMBER_OF_PROCESSES = torch.cuda.device_count()
print(f'Max number of processes: {MAX_NUMBER_OF_PROCESSES}')
print(f'Workspace path: {WORKSPACE_PATH}')




with open('config/config_spectra.yml') as file:
    config_spectra = yaml.load(file, Loader=yaml.FullLoader)

default_config = {
    'lr': 1e-3,
    'hidden_channels': 256,
    'num_layers': 3,
    'num_timesteps': 3,
    'dropout': 0.05,
    #'seed': 42,
    'num_workers': 0,
    'ep_train': 2,
    'run_id': None
}

def parse_args():
    "Overriding default parameters"
    argparser = argparse.ArgumentParser(description='Process hyperparameters')
    #argparser.add_argument('--batch_size', type=int, default=default_config['batch_size'], help='Batch size')
    argparser.add_argument('--lr', type=float, default=default_config['lr'], help='Learning rate')
    argparser.add_argument('--hidden_channels', type=int, default=default_config['hidden_channels'], help='Hidden channels')
    argparser.add_argument('--num_layers', type=int, default=default_config['num_layers'], help='Number of layers')
    argparser.add_argument('--num_timesteps', type=int, default=default_config['num_timesteps'], help='Number of timesteps')
    argparser.add_argument('--dropout', type=float, default=default_config['dropout'], help='Dropout')
    argparser.add_argument('--num_workers', type=int, default=default_config['num_workers'], help='Number of workers')
    argparser.add_argument('--run_id', type=str, default=default_config['run_id'], help='Run ID for resmuming training')
    argparser.add_argument('--ep_train', type=int, default=default_config['ep_train'], help='Number of epochs to train')
    args = argparser.parse_args()
    for arg in vars(args):
        default_config[arg] = args.__dict__[arg]
    return

def data_to_workspace():
    if not os.path.exists(WORKSPACE_PATH):
        os.makedirs(WORKSPACE_PATH, exist_ok=True)
    process_dirs = [d for d in os.listdir(WORKSPACE_PATH) if d.startswith('process_')]
    i = -1
    if len(process_dirs) != 0:
        # For each process directory, look for a status.yml file
        for process_dir in process_dirs:
            i += 1
            status_file = os.path.join(WORKSPACE_PATH, process_dir, 'status.yml')
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = yaml.load(f, Loader=yaml.FullLoader)
                if status['status'] == 'running':
                    continue
                else:
                    i -= 1
                    break
            else:
                i -= 1
                break        
        
    if i < MAX_NUMBER_OF_PROCESSES:
        # create a new process directory
        process_dir = os.path.join(WORKSPACE_PATH, f'process_{i+1}')
        if not os.path.exists(process_dir):
            os.makedirs(os.path.join(WORKSPACE_PATH, f'process_{i+1}'), exist_ok=True)
            os.makedirs(os.path.join(WORKSPACE_PATH, f'process_{i+1}', 'plots'), exist_ok=True)
        # create a new status file
        with open(os.path.join(process_dir, 'status.yml'), 'w') as f:
            yaml.dump({'status': 'running'}, f)
        os.makedirs(process_dir, exist_ok=True)
        script_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(os.path.join(process_dir, 'data')):
            original_data = os.path.join(script_path, 'data')
            print(f'Copying data from {original_data} to {process_dir}')
            print(f'Could take a while depending on the size of the data. Please wait.')
            os.system(f'cp -r {original_data} {process_dir}')
    else:
        return None
        
    return process_dir
    

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def mse_loss(y_true, y_pred):
    loss = F.mse_loss(y_true, y_pred)
    return loss

def train(model, device, optimizer, loader, loss_function):
    model.train()
    total_loss = total_examples = 0
    #batch_loss = []
    with tqdm(total=len(loader)) as bar:
        for i, data in enumerate(loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            y = data.y/torch.max(data.y, dim=1, keepdim=True)[0]
            y = torch.sqrt(y)
            
            loss = loss_function(out, y)
            loss.backward()
            optimizer.step()
            
            out = torch.pow(out, 2)
            y = torch.pow(y, 2)
            
            loss = F.mse_loss(out, y, reduction='mean')
            total_loss += loss.item() * data.num_graphs
            total_examples += data.num_graphs
            
            bar.set_description('Train')
            bar.update(1)
            bar.set_postfix(loss=loss.item(), lr=get_lr(optimizer))
    return total_loss / total_examples

@torch.no_grad()
def validation(model, device, optimizer, loader):
    model.eval()
    total_loss = total_examples = 0
    with tqdm(total=len(loader)) as bar:
        for data in loader:
            data = data.to(device)
            # Read batch size from the data
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            out = torch.pow(out, 2)
            y = data.y/torch.max(data.y, dim=1, keepdim=True)[0]
            
            loss = F.mse_loss(out, y, reduction='mean').item()
            total_loss += loss * data.num_graphs
            total_examples += data.num_graphs
            
            bar.set_description('Validation')
            bar.update(1)
            bar.set_postfix(loss=loss, lr=get_lr(optimizer))
    return total_loss / total_examples

@torch.no_grad()
def test(model, device, loader, config_spectra, process_dir):
    model.eval()
    from plot_utils import make_subplot, make_density_plot
    predictions = {'smiles': [], 'y': [], 'y_pred': [], 'loss': []}
    with tqdm(total=len(loader)) as bar:

        for data in loader:
            data = data.to(device)
            # Read batch size from the data
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            out = torch.pow(out, 2)
            
            y = data.y/torch.max(data.y, dim=1, keepdim=True)[0]
            
            loss = F.mse_loss(out, y, reduction='none')
            loss = torch.mean(loss, dim=1)
            loss = loss.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            out = out.cpu().detach().numpy()
            smiles = data.smiles
            # Group loss, y and out to iterate over associated rows
            for i in range(y.shape[0]):
                predictions['smiles'].append(smiles[i])
                predictions['y'].append(y[i])
                predictions['y_pred'].append(out[i])
                predictions['loss'].append(loss[i])
            # Find worst prediction
            bar.set_description(f'Test batch')
            bar.update(1)
    predictions = pd.DataFrame(predictions)
    fig = make_subplot(predictions, config_spectra)
    density_fig_name = make_density_plot(predictions, config_spectra, process_dir)
    
    return fig, density_fig_name

def main(default_config):
    process_dir = data_to_workspace()
    if process_dir is None:
        print('No hardware available to run the process. Exiting.')
        return
    else:
        print(f'Process directory: {process_dir}')
        # REad ending integer from process_dir
        gpu_id = int(process_dir.split('_')[-1])
        print(f'GPU ID: {gpu_id}')
        try:
            device = torch.device(f'cuda:{gpu_id}')
        except:
            print('GPU not found. Exiting.')
            # Change the status of the process to finished
            with open(os.path.join(process_dir, 'status.yml'), 'w') as f:
                yaml.dump({'status': 'finished'}, f)
                return
        

    #try:
    if default_config['run_id'] is not None:
        try:
            run_wandb = wandb.init(project=params.WANDB_PROJECT,
                        entity=params.ENTITY,
                        config=default_config,
                        job_type='traning',
                        id=default_config['run_id'],
                        resume='must')
        except:
            ValueError('Run ID not found. Exiting.')
    else:
        run_wandb = wandb.init(project=params.WANDB_PROJECT,
                        entity=params.ENTITY,
                        config=default_config,
                        job_type='traning')
    
    config = wandb.config
    checkpoint_file = run_wandb.id
    checkpoint_file = os.path.join('models', f'checkpoint_{checkpoint_file}.pt')        
    
    model = AttentiveFP(
        in_channels=26,
        hidden_channels=config["hidden_channels"],
        out_channels=config_spectra["out_dim"],
        edge_dim=14,
        num_layers=config["num_layers"],
        num_timesteps=config["num_timesteps"],
        dropout=config["dropout"]
    )
            
    path_train = osp.join(process_dir, 'data', 'train', 'data')
    path_train_split = osp.join(path_train, 'raw', 'data', 'split_dict.pt')
    path_val = osp.join(process_dir, 'data', 'val', 'data')
    path_val_split = osp.join(path_val, 'raw', 'data', 'split_dict.pt')
    path_test = osp.join(process_dir, 'data', 'test', 'data')
    path_test_split = osp.join(path_test, 'raw', 'data', 'split_dict.pt')
    
    split = [config_spectra['split_train'], config_spectra['split_val'], config_spectra['split_test']]
    GenSplit(root= path_train_split, split=split)
    data_train = DatasetAttentiveFP(root=path_train, split='train', one_hot=config_spectra['one_hot'], config=config_spectra)
    GenSplit(root= path_val_split, split=split)
    data_val = DatasetAttentiveFP(root=path_val, split='val', one_hot=config_spectra['one_hot'], config=config_spectra)
    GenSplit(root= path_test_split, split=split)
    data_test = DatasetAttentiveFP(root=path_test, split='test', one_hot=config_spectra['one_hot'], config=config_spectra)
    
    batch_size = find_batch_size(model, device, gpu_id, data_train)
            
    train_loader= DataLoader(data_train, batch_size=batch_size, 
                            shuffle=True, num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, 
                            shuffle=True, num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, 
                            shuffle=True, num_workers=config['num_workers'], pin_memory=True, drop_last=True)

    import warnings
    warnings.filterwarnings("ignore",
                        "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*")
    loss_function = torch.compile(mse_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=4.e-4)
    model = model.to(device)
    model = torch.compile(model, dynamic=True)
    print(model)
    ep_train = config['ep_train']
    if wandb.run.resumed:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_mse = checkpoint['loss']
        best_model = model
        best_optimizer = optimizer
        ep_train = ep_train + epoch
        print(f'Resuming training from epoch {epoch}')
        print(f'Loaded model from checkpoint: {checkpoint_file}')
    else:
        epoch = 0
        best_val_mse = 999999
        best_model = None
        best_optimizer = None
    patience = 10
    no_improvement = 1
    best_train_mse = 999999
    while epoch < ep_train:
        epoch += 1 
        train_mse = train(model, device, optimizer, train_loader, loss_function)
        val_mse = validation(model, device, optimizer, val_loader)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
            best_optimizer = optimizer
            torch.save({'epoch': epoch,
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': best_optimizer.state_dict(),
                        'loss': best_val_mse},
                        checkpoint_file)
            patience = 10
            fig, density_fig_name = test(model, device, test_loader, config_spectra, process_dir)
            run_wandb.log({"Train MSE": train_mse, 
                        "Validation MSE": val_mse, 
                        "Learning rate": get_lr(optimizer),
                        "Spectra": fig,
                        "Density plot": wandb.Image(osp.join(process_dir, 'plots', density_fig_name))})
            
        else:
            run_wandb.log({"Train MSE": train_mse, 
                        "Validation MSE": val_mse, 
                        "Learning rate": get_lr(optimizer)})
            patience -= 1
            if patience == 0:
                break
        if train_mse < best_train_mse:
            best_train_mse = train_mse
            no_improvement = 1
        else:
            no_improvement += 1
            if no_improvement == 5:
                # Update learning rate of the optimizer. Reduce by half.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2

    run_wandb.finish()
    # Change the status of the process to finished
    with open(os.path.join(process_dir, 'status.yml'), 'w') as f:
        yaml.dump({'status': 'finished'}, f)
    #except:
    #    # Change the status of the process to finished
    #    with open(os.path.join(process_dir, 'status.yml'), 'w') as f:
    #        yaml.dump({'status': 'finished'}, f)
 
if __name__ == "__main__":
   parse_args()
   main(default_config)