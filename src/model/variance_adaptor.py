import torch
import numpy as np
from torch import Tensor
from torch.nn import Module, Parameter, Embedding
from .components.variance_predictor import VariancePredictor
from .components.length_regulator import LengthRegulator
from .utils import load_json, get_mask_from_lengths

class VarianceAdaptor(Module):
    
    # In this project, pitch and energy will be processed in linear scale and predicted in frame level
    def __init__(self, d_in: int, conv_chans: int = 256, kernel_size: int = 3, 
                 dropout:float=0.5, n_bins: int = 256) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.duration_predictor = VariancePredictor(
            d_in=d_in, conv_chans=conv_chans, kernel_size=kernel_size, dropout=dropout
        )
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(
            d_in=d_in, conv_chans=conv_chans, kernel_size=kernel_size, dropout=dropout
        )
        self.energy_predictor = VariancePredictor(
            d_in=d_in, conv_chans=conv_chans, kernel_size=kernel_size, dropout=dropout
        )
        #TODO: put file name in config.yaml
        stats = load_json("data/final_LJSpeech/stats.json") 
        pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
        energy_min, energy_max, energy_mean, energy_std = stats["energy"]
        # bins for pitch and energy (linear scale), n_bins-1 so that when bucketize we have n_bins
        self.pitch_bins = Parameter(
            torch.linspace(pitch_min, pitch_max, n_bins-1), requires_grad=False
        )
        self.energy_bins = Parameter( 
            torch.linspace(energy_min, energy_max, n_bins-1), requires_grad=False
        ) 
        
        # According to the paper, there is a embedding layer for 
        # target pitch and energy to learn pitch and energy
        self.pitch_embedding = Embedding(n_bins, d_in)
        self.energy_embedding = Embedding(n_bins, d_in)
    
            
    def forward(self, x: Tensor, dur_trg: Tensor, pitch_trg: Tensor, energy_trg: Tensor = None, src_mask: Tensor = None, 
                mel_mask: Tensor = None, max_dur: int = None, p_control: float = 1.0, e_control: float = 1.0, 
                d_control: float = 1.0):
        log_dur_pred = self.duration_predictor(x, src_mask)
        dur_rounded = torch.clamp(torch.round(torch.exp(log_dur_pred) - 1) * d_control, min=0)
        
        if dur_trg != None: 
            x, mel_durs = self.length_regulator(x, dur_trg, max_dur)
            dur_rounded = dur_trg
        else: 
            x, mel_durs = self.length_regulator(x, dur_rounded, max_dur)
            
        mel_mask = get_mask_from_lengths(mel_durs, self.device) # get new mask after length regulation
        pitch_pred = self.pitch_predictor(x, mel_mask) * p_control
        if pitch_trg != None: 
            pitch_emb = self.pitch_embedding(torch.bucketize(pitch_trg, self.pitch_bins))
        else: 
            pitch_emb = self.pitch_embedding(torch.bucketize(pitch_pred, self.pitch_bins))
        x = x + pitch_emb
        
        energy_pred = self.energy_predictor(x, mel_mask) * e_control
        if energy_trg != None: 
            energy_emb = self.energy_embedding(torch.bucketize(pitch_trg, self.pitch_bins))
        else: 
            energy_emb = self.energy_embedding(torch.bucketize(energy_pred, self.energy_bins))
        x = x + energy_emb
        
        return (x, mel_mask, log_dur_pred, dur_rounded, pitch_pred, pitch_emb, energy_pred, energy_emb)        
        