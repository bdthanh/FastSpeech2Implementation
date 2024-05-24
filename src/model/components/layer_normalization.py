import torch 
from torch import nn, Tensor
from torch.nn import Module

class LayerNorm(Module):
    
    def __init__(self, d_hidden: int = 256, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.d_hidden = d_hidden
        self.alpha = nn.parameter.Parameter(torch.ones(d_hidden), requires_grad=True)
        self.beta = nn.parameter.Parameter(torch.zeros(d_hidden), requires_grad=True)
        
        
    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm_x = (x - mean) / (std + self.eps)
        return self.alpha * norm_x + self.beta
        