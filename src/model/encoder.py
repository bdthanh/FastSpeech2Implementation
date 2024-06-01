from copy import deepcopy
from torch import Tensor
from torch.nn import Dropout, Module, ModuleList
from .components.phoneme_embedding import PhonemeEmbedding
from .components.layer_normalization import LayerNorm
from .components.multi_head_attention import MultiHeadAttention
from .components.position_wise_feed_forward import PositionWiseFeedForward
from .components.positional_encoding import PositionalEncoding, get_positional_encoding

class Encoder(Module):
  
    def __init__(self, phoneme_size: int, d_hidden: int = 256, conv_chans: int = 1024, kernel_size: int = 9, 
                 n_heads: int = 2, n_layers: int = 4, dropout: float = 0.1, eps: float = 1e-6, max_seq_len: int = 1000) -> None:
        super().__init__()
        self.embedding = PhonemeEmbedding(phoneme_size=phoneme_size, d_hidden=d_hidden)
        # self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_hidden)
        single_layer = EncoderLayer(
            d_hidden=d_hidden, conv_chans=conv_chans, n_heads=n_heads, kernel_size=kernel_size, dropout=dropout, eps=eps
        )
        self.encoder_layers = ModuleList([deepcopy(single_layer) for _ in range(n_layers)])
        
        
    def forward(self, src: Tensor, mask: Tensor):
        x = self.embedding(src)
        _, seq_len, _ = x.size()
        device = x.device
        x = x + get_positional_encoding(seq_len, x.size(-1)).to(device)
        attn_mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, -1, seq_len, -1)
        normal_mask = mask.unsqueeze(-1)
        for layer in self.encoder_layers:
            x = layer(x, normal_mask, attn_mask)
            
        return x  
      
      
class EncoderLayer(Module):
  
    def __init__(self, d_hidden: int = 256, conv_chans: int = 1024, n_heads: int = 2, kernel_size: int = 9,
                 dropout: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_hidden=d_hidden, n_heads=n_heads, dropout=dropout)
        self.self_attn_dropout = Dropout(p=dropout)
        self.self_attn_layer_norm = LayerNorm(d_hidden=d_hidden, eps=eps)
        
        self.feed_fwd = PositionWiseFeedForward(d_in=d_hidden, conv_chans=conv_chans, kernel_size=kernel_size, dropout=dropout)
        self.feed_fwd_dropout = Dropout(p=dropout)
        self.feed_fwd_layer_norm = LayerNorm(d_hidden=d_hidden, eps=eps)
        
        
    def forward(self, x: Tensor, normal_mask: Tensor, attn_mask: Tensor):
        _x = self.self_attn(x, x, x, attn_mask)
        x = self.self_attn_layer_norm(x + self.self_attn_dropout(_x))
        x = x.masked_fill(normal_mask==0, 0)
        
        _x = self.feed_fwd(x)
        x = self.feed_fwd_layer_norm(x + self.feed_fwd_dropout(_x))
        x = x.masked_fill(normal_mask==0, 0)
        
        return x
        