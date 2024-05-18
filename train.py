import argparse
import os
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
from torch import nn 
from torch.utils.data import DataLoader
from src.optimizer.optimizer import ScheduledOptim, get_optimizer
from src.dataset.parallel_dataset import ParallelDataset
from src.dataset.symbol_vocabulary import SymbolVocabulary
from src.model.fastspeech2 import FastSpeech2, get_fastspeech2
from src.model.fastspeech2_loss import FastSpeech2Loss
from src.utils import choose_device, load_config, create_if_missing_folder, is_file_exist, get_num_params, get_mask_from_lengths

def save_checkpoint(path, model: FastSpeech2, optimizer:ScheduledOptim, epoch, global_step):
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

def unpack_batch(batch, device):
    ids = batch['ids']
    raw_texts = batch['raw_texts']
    speakers = batch['speakers']
    phonemes = torch.from_numpy(batch['phonemes']).long().to(device)
    phonemes_lens = torch.from_numpy(batch['phonemes_lens']).to(device)
    max_phoneme_len = batch['max_phoneme_len']
    mels = torch.from_numpy(batch['mels']).float().to(device)
    mel_lens = torch.from_numpy(batch['mel_lens']).to(device)
    max_mel_len = batch['max_mel_len']
    pitches = torch.from_numpy(batch['pitches']).float().to(device)
    energies = torch.from_numpy(batch['energies']).float().to(device)
    durations = torch.from_numpy(batch['durations']).long().to(device)

    return ids, raw_texts, speakers, phonemes, phonemes_lens, max_phoneme_len, mels, mel_lens, max_mel_len, pitches, energies, durations

def epoch_eval(model: FastSpeech2, global_step: int, epoch: int, val_dataloader: DataLoader, device):
    pass

def train(config):
    device = choose_device()
    symbol_vocab, train_loader, valid_loader = get_ds(config)
    model = get_fastspeech2(config, len(symbol_vocab))
    optimizer = get_optimizer(model, config, cur_step=0)
    initial_epoch, global_step = 0, 0
    model, optimizer, initial_epoch, global_step = load_checkpoint_if_exists(
        config['path']['checkpoint_last'], model, optimizer
    )
    num_params = get_num_params(model)
    print(f"Number of FastSpeech2 parameter is: {num_params}")
    loss_func = FastSpeech2Loss()
    # wandb.define_metric("global_step")
    # wandb.define_metric("validation/*", step_metric="global_step")
    # wandb.define_metric("train/*", step_metric="global_step")
    
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
            ids, raw_texts, speakers, phonemes, phonemes_lens, max_phoneme_len, mel_trg, mel_lens, max_mel_len, pitch_trg, energy_trg, dur_trg = unpack_batch(batch, device)
            src_mask = get_mask_from_lengths(phonemes_lens, device, max_phoneme_len)
            mel_mask = get_mask_from_lengths(mel_lens, device, max_mel_len)
            #need 2 mel_mask for pred duration
            mel_pred, mel_mask, postnet_mel_pred, log_dur_pred, dur_rounded, pitch_pred, pitch_emb, energy_pred, energy_emb = model(
                phonemes, src_mask, mel_mask, pitch_trg, energy_trg
            )    
            total_loss, mel_loss, mel_postnet_loss, dur_loss, pitch_loss, energy_loss = loss_func(
                mel_trg, dur_trg, pitch_trg, energy_trg, mel_pred, postnet_mel_pred, log_dur_pred, pitch_pred, energy_pred, src_mask, mel_mask
            )
            total_loss = total_loss / grad_acc_step
            
            # wandb.log({'train/total_loss': total_loss.item(), 'global_step': global_step})
            # wandb.log({'train/mel_loss': mel_loss.item(), 'global_step': global_step})
            # wandb.log({'train/mel_postnet_loss': mel_postnet_loss.item(), 'global_step': global_step})
            # wandb.log({'train/dur_loss': dur_loss.item(), 'global_step': global_step})
            # wandb.log({'train/pitch_loss': pitch_loss.item(), 'global_step': global_step})
            # wandb.log({'train/energy_loss': energy_loss.item(), 'global_step': global_step})
            
            total_loss.backward()
            if global_step % grad_acc_step == 0:
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                # Update weights
                optimizer.step_and_update_lr()
            global_step+=1
        epoch_eval(model, global_step, epoch, valid_loader, device)
        save_checkpoint(config['path']['checkpoint_last'], model, optimizer, epoch, global_step)
            

if __name__ == '__main__':
    current_file_path = Path(__file__).resolve() 
    current_dir = current_file_path.parent 
    config_path = current_dir / 'config' / 'config.yaml'
    config = load_config(config_path)
    create_if_missing_folder(config['path']['checkpoint_path'])
    # wandb.init(project='fastspeech2', config=config)
    train(config)
    # wandb.finish()
