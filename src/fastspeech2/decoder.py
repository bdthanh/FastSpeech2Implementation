from copy import deepcopy
from torch import Tensor
from torch.nn import Dropout, Module, ModuleList
from .components.phoneme_embedding import PhonemeEmbedding
from .components.layer_normalization import LayerNorm
from .components.multi_head_attention import MultiHeadAttention
from .components.position_wise_feed_forward import PositionWiseFeedForward
from .components.positional_encoding import PositionalEncoding 

class Decoder(Module):
  
    def __init__(self, d_hidden: int = 256, conv_chans: int = 1024, n_heads: int = 2, n_layers: int = 4, 
                 dropout: float = 0.1, eps: float = 1e-9, max_seq_len: int = 1000) -> None:
        super().__init__()
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_hidden)
        single_layer = DecoderLayer(d_hidden=d_hidden, conv_chans=conv_chans, n_heads=n_heads, dropout=dropout, eps=eps)
        self.decoder_layers = ModuleList([deepcopy(single_layer) for _ in range(n_layers)])
        
        
    def forward(self, src: Tensor, mask: Tensor):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, mask)
            
        return x  
      
      
class DecoderLayer(Module):
  
    def __init__(self, d_hidden: int = 256, conv_chans: int = 1024, n_heads: int = 2, dropout: float = 0.1, 
                 eps: float = 1e-9) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_hidden=d_hidden, n_heads=n_heads, dropout=dropout)
        self.self_attn_dropout = Dropout(p=dropout)
        self.self_attn_layer_norm = LayerNorm(d_hidden=d_hidden, eps=eps)
        
        self.feed_fwd = PositionWiseFeedForward(d_in=d_hidden, conv_chans=conv_chans, dropout=dropout)
        self.feed_fwd_dropout = Dropout(p=dropout)
        self.feed_fwd_layer_norm = LayerNorm(d_hidden=d_hidden, eps=eps)
        
    def forward(self, x: Tensor, mask: Tensor):
        _x = self.self_attn(x, x, x, mask)
        x = self.self_attn_layer_norm(x + self.self_attn_dropout(_x))
        
        _x = self.feed_fwd(x)
        x = self.feed_fwd_layer_norm(x + self.feed_fwd_dropout(_x))
        
        return x 
        