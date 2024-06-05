import torch
from torch.nn import MSELoss, L1Loss, Module

# The loss is calculated based on info from Tacotron 2, FastSpeech and FastSpeech2 paper
class FastSpeech2Loss(Module):

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = MSELoss()
        self.mae_loss = L1Loss()

        
    def forward(self, mel_trg, dur_trg, pitch_trg, energy_trg, mel_pred, mel_postnet_pred,
                log_dur_pred, pitch_pred, energy_pred, src_mask, mel_mask):
        src_mask = ~src_mask
        mel_masks = ~mel_mask
        log_dur_trg = torch.log(dur_trg.float() + 1)
        mel_trg.requires_grad = False 
        log_dur_trg.requires_grad = False 
        pitch_trg.requires_grad = False 
        energy_trg.requires_grad = False 
        mel_trg = mel_trg[:, : mel_mask.shape[1], :]
        mel_mask = mel_mask[:, :mel_mask.shape[1]]
        
        pitch_pred = pitch_pred.masked_select(mel_mask)
        pitch_trg = pitch_trg.masked_select(mel_mask)
        energy_pred = energy_pred.masked_select(mel_mask)
        energy_trg = energy_trg.masked_select(mel_mask)
        log_dur_pred = log_dur_pred.masked_select(src_mask)
        log_dur_trg = log_dur_trg.masked_select(src_mask)
        mel_pred = mel_pred.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet_pred = mel_postnet_pred.masked_select(mel_mask.unsqueeze(-1))
        mel_trg = mel_trg.masked_select(mel_mask.unsqueeze(-1))
        
        mel_loss = self.mae_loss(mel_pred, mel_trg)
        mel_postnet_loss = self.mae_loss(mel_postnet_pred, mel_trg)

        pitch_loss = self.mse_loss(pitch_pred, pitch_trg)
        energy_loss = self.mse_loss(energy_pred, energy_trg)
        dur_loss = self.mse_loss(log_dur_pred, log_dur_trg)

        total_loss = mel_loss + mel_postnet_loss + dur_loss + pitch_loss + energy_loss
        
        return total_loss, mel_loss, mel_postnet_loss, dur_loss, pitch_loss, energy_loss
        