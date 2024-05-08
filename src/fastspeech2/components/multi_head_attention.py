import torch
from math import sqrt
from torch import Tensor
from torch.nn import Module, Dropout, Softmax, Linear

class MultiHeadAttention(Module):
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.sdp_attn = ScaledDotProductAttention(dropout=dropout)
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_concat = Linear(d_model, d_model)
        
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q, k, v = self._split(q), self._split(k), self._split(v)
        attn = self.sdp_attn(q, k, v, mask=mask)
        attn = self._concat(attn)
        attn = self.linear_concat(attn)
        
        return attn
      
      
    def _split(self, x: Tensor):
        batch_size, length, _ = x.size()
        assert self.d_model % self.n_heads == 0, "Division of d_model to n_heads has remainder"
        d_head = self.d_model // self.n_heads
        x = x.view(batch_size, length, self.n_heads, d_head).transpose(1, 2)
      
        return x   
      
      
    def _concat(self, x: Tensor):
        batch_size, _, _, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return x
        
        
class ScaledDotProductAttention(Module):
    
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(p=dropout)
    
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        d_k = k.size()[-1]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / sqrt(d_k)
        if mask is not None: 
            attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
        attn_scores = self.softmax(attn_scores)
        v = torch.matmul(self.dropout(attn_scores), v)
        
        return v
    