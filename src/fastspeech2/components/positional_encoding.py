import torch
from torch import Tensor
from torch.nn import Module

class PositionalEncoding(Module):
  
    def __init__(self, n_pos: int, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = n_pos
        positional_encoding = torch.zeros(n_pos, d_model)
        positional_encoding.requires_grad = False
        pos = torch.arange(0, n_pos).unsqueeze(dim=1)
        positional_encoding[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        positional_encoding[:, 1::2] = torch.cos(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        positional_encoding = positional_encoding.unsqueeze(0)  
        self.register_buffer('positional_encoding', positional_encoding)        
        
        
    def forward(self, x: Tensor):
        _, seq_len, _ = x.size() # [batch_size, seq_len, d_model]
        x = x + self.positional_encoding[:, :seq_len]
        
        return x
