import argparse
import os
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.optimizer.optimizer import ScheduledOptim, get_optimizer
from src.dataset.parallel_dataset import ParallelDataset
from src.dataset.symbol_vocabulary import SymbolVocabulary
from src.model.fastspeech2 import FastSpeech2, get_fastspeech2
from src.model.fastspeech2_loss import FastSpeech2Loss
from src.utils import choose_device, load_config, create_if_missing_folder, is_file_exist, get_num_params, get_mask_from_lengths

def save_checkpoint(path, model, optimizer, epoch, global_step):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    
def load_checkpoint_if_exists(path, model: FastSpeech2, optimizer: ScheduledOptim):
    initial_epoch, global_step = 0, 0
    if is_file_exist(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], global_step)

        print(f'Loaded checkpoint from epoch {initial_epoch}')
    return model, optimizer, initial_epoch, global_step

def get_ds(config):
    print("Getting dataset..........")
    symbol_vocab = SymbolVocabulary()
    train_ds = ParallelDataset(symbol_vocab, "train.txt", config)
    valid_ds = ParallelDataset(symbol_vocab, "valid.txt", config)
    train_loader = DataLoader(
        train_ds, batch_size=config['optimizer']['batch_size'], shuffle=True, collate_fn=train_ds.collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=1, shuffle=True, collate_fn=valid_ds.collate_fn
    )
    return symbol_vocab, train_loader, valid_loader

def train(config):
    device = choose_device()
    symbol_vocab, train_loader, valid_loader = get_ds(config)
    model = get_fastspeech2(config)
    optimizer = get_optimizer(model, config, cur_step=0)
    initial_epoch, global_step = 0, 0
    model, optimizer, initial_epoch, global_step = load_checkpoint_if_exists(
        config['path']['checkpoint_last'], model, optimizer
    )
    num_params = get_num_params(model)
    print(f"Number of FastSpeech2 parameter is: {num_params}")
    loss = FastSpeech2Loss()
    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    
    grad_acc_step = config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = config["optimizer"]["grad_clip_thresh"]
    total_step = config["step"]["total_step"]
    save_step = config["step"]["save_step"]
    synth_step = config["step"]["synth_step"]
    val_step = config["step"]["val_step"]
    num_epochs = config["step"]["num_epochs"]
    
    for epoch in range(initial_epoch, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iter = tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}", total=len(train_loader))
        for batch in batch_iter:
            optimizer.zero_grad()            
    

if __name__ == '__main__':
    current_file_path = Path(__file__).resolve() 
    current_dir = current_file_path.parent 
    config_path = current_dir / 'config' / 'config.yaml'
    config = load_config(config_path)
    create_if_missing_folder(config['path']['checkpoint_dir'])
    wandb.init(project='fastspeech2', config=config)
    train(config)
    wandb.finish()