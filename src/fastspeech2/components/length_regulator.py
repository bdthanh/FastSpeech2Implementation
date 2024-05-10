import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

class LengthRegulator(Module):
    #TODO: To be implemented
    def __init__(self) -> None:
        super().__init__() 
    
    
    def forward(self, x: Tensor, x_pred_dur, max_dur: int = None):
        #x: [B, seq_len, dim]
        #x_len_ratio: [B, seq_len]
        output, mel_dur = [], []
        for inner_x, pred_dur in zip(x, x_pred_dur):
            out = []
            for i, inner_x_item in enumerate(inner_x):
                out.extend([inner_x_item] * pred_dur[i])
                out = torch.cat(out, 0)
            output.append(out)
            mel_dur.append(out.size(0))
        if max_dur is not None:
            output = [torch.cat([mel, torch.zeros(max_dur - mel.size(0), mel.size(1))], 0) for mel in output]
        else: 
            pass
        return output, mel_dur
    
def _pad(x, mel_max_dur=None):
    if mel_max_dur:
        mel_dur = mel_max_dur
    else: # max_dur in the batch
        mel_dur = max([x[i].size(0) for i in range(len(x))])

    out_list = list()
    for i, batch in enumerate(x):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, mel_dur - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, mel_dur - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded