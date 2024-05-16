from copy import deepcopy
from torch import Tensor
from torch.nn import Linear, Dropout, Module, ModuleList
from .components.phoneme_embedding import PhonemeEmbedding
from .components.layer_normalization import LayerNorm
from .components.multi_head_attention import MultiHeadAttention
from .components.position_wise_feed_forward import PositionWiseFeedForward
from .components.positional_encoding import PositionalEncoding 

class Decoder(Module):
  
    def __init__(self, d_hidden: int = 256, conv_chans: int = 1024, n_mel_chans: int = 80, kernel_size: int = 9, 
                 n_heads: int = 2, n_layers: int = 4, dropout: float = 0.1, eps: float = 1e-6, max_seq_len: int = 1000) -> None:
        #TODO: Check pos encoding max_seq_len in FastSpeech 2
        super().__init__()
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_hidden)
        single_layer = DecoderLayer(d_hidden=d_hidden, conv_chans=conv_chans, kernel_size=kernel_size, n_heads=n_heads, dropout=dropout, eps=eps)
        self.decoder_layers = ModuleList([deepcopy(single_layer) for _ in range(n_layers)])
        self.linear = Linear(d_hidden, n_mel_chans)
        
        
    def forward(self, x: Tensor, mask: Tensor):
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, mask)
            
        return self.linear(x)
      
      
class DecoderLayer(Module):
  
    def __init__(self, d_hidden: int = 256, conv_chans: int = 1024, kernel_size: int = 9, n_heads: int = 2, 
                 dropout: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_hidden=d_hidden, n_heads=n_heads, dropout=dropout)
        self.self_attn_dropout = Dropout(p=dropout)
        self.self_attn_layer_norm = LayerNorm(d_hidden=d_hidden, eps=eps)
        
        self.feed_fwd = PositionWiseFeedForward(d_in=d_hidden, conv_chans=conv_chans, kernel_size=kernel_size, dropout=dropout)
        self.feed_fwd_dropout = Dropout(p=dropout)
        self.feed_fwd_layer_norm = LayerNorm(d_hidden=d_hidden, eps=eps)
        
        
    def forward(self, x: Tensor, mask: Tensor):
        _x = self.self_attn(x, x, x, mask)
        x = self.self_attn_layer_norm(x + self.self_attn_dropout(_x))
        x = x.masked_fill(mask==0, 0)
        
        _x = self.feed_fwd(x)
        x = self.feed_fwd_layer_norm(x + self.feed_fwd_dropout(_x))
        x = x.masked_fill(mask==0, 0)
        
        return x
        