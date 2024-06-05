import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

class LengthRegulator(Module):
   
    def __init__(self) -> None:
        super().__init__() 
    
    
    def forward(self, x: Tensor, x_pred_dur, max_dur: int = None):
        #x: [B, seq_len, dim]
        #x_pred_dur: [B, seq_len]
        output, mel_durs = [], []
        for inner_x, pred_dur in zip(x, x_pred_dur): # inner_x: [seq_len, dim], pred_dur: [seq_len]
            out = []
            for i, inner_x_item in enumerate(inner_x): 
                out.append(inner_x_item.repeat(int(pred_dur[i].item()), 1))
            out = torch.cat(out, 0)
            output.append(out)
            mel_durs.append(out.size(0))
        if max_dur is not None:
            output = _pad(output, max_dur)
        else: # pad to the longest length in the batch
            output = _pad(output)
        return output, torch.LongTensor(mel_durs)
    
    
def _pad(x, mel_max_dur=None):
    if mel_max_dur:
        mel_dur = mel_max_dur
    else: # max_dur in the batch
        mel_dur = max([x[i].size(0) for i in range(len(x))])

    out_list = []
    for batch in x:
        #TODO: Handle cases for mel_dur - batch.size(0) < 0
        one_batch_padded = F.pad(
            batch, pad=(0, 0, 0, mel_dur - batch.size(0)), mode="constant", value=0.0
        )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded