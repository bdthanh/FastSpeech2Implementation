import torch
from torch.nn import MSELoss, L1Loss, Module

# The loss is calculated based on info from Tacotron 2, FastSpeech and FastSpeech2 paper
class FastSpeech2Loss(Module):
    #TODO: To be implemented
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = MSELoss()
        self.mae_loss = L1Loss()

        
    def forward(self, mel_trg, dur_trg, pitch_trg, energy_trg, mel_pred, mel_postnet_pred,
                log_dur_pred, pitch_pred, energy_pred, src_mask, mel_mask):
        log_dur_trg = torch.log(dur_trg.float() + 1)
        mel_trg.requires_grad = False 
        log_dur_trg.requires_grad = False 
        pitch_trg.requires_grad = False 
        energy_trg.requires_grad = False 
        pass