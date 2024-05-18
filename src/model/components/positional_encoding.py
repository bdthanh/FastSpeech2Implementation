import torch
from torch import Tensor
from torch.nn import Module

def get_positional_encoding(seq_len: int, d_model: int) -> Tensor:
    positional_encoding = torch.zeros(seq_len, d_model)
    positional_encoding.requires_grad = False
    pos = torch.arange(0, seq_len).unsqueeze(dim=1)
    positional_encoding[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
    positional_encoding[:, 1::2] = torch.cos(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
    return positional_encoding.unsqueeze(0)


class PositionalEncoding(Module):
  
    def __init__(self, seq_len: int = 1000, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        positional_encoding = torch.zeros(seq_len, d_model)
        positional_encoding.requires_grad = False
        pos = torch.arange(0, seq_len).unsqueeze(dim=1)
        positional_encoding[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        positional_encoding[:, 1::2] = torch.cos(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        positional_encoding = positional_encoding.unsqueeze(0)  
        self.register_buffer('positional_encoding', positional_encoding)        
        
        
    def forward(self, x: Tensor):
        _, seq_len, _ = x.size() # [batch_size, seq_len, d_model]
        x = x + self.positional_encoding[:, :seq_len]
        
        return x
