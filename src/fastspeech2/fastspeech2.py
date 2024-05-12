from torch import Tensor
from torch.nn import Module, init
from .encoder import Encoder
from .variance_adaptor import VarianceAdaptor
from .decoder import Decoder
from .postnet import PostNet

class FastSpeech2(Module):
    
    def __init__(self, phoneme_size, max_seq_len, enc_hidden, enc_heads, enc_layers, dec_hidden, dec_heads, 
                 dec_layers, ed_conv_chans, ed_kernel_size, enc_dropout, dec_dropout, var_conv_chans, var_kernel_size, 
                 n_bins, var_dropout, n_mel_chans, pn_conv_chans, pn_kernel_size, pn_layers, 
                 pn_dropout, pn_act_fn, eps = 1e-6) -> None:
        super().__init__()
        self.encoder = Encoder(
            phoneme_size=phoneme_size, d_hidden=enc_hidden, conv_chans=ed_conv_chans, kernel_size=ed_kernel_size,
            n_heads=enc_heads, n_layers=enc_layers, dropout=enc_dropout, eps=eps, max_seq_len=max_seq_len
        )
        
        self.variance_adaptor = VarianceAdaptor(
            d_in=enc_hidden, conv_chans=var_conv_chans, kernel_size=var_kernel_size, dropout=var_dropout, n_bins= n_bins
        )
        
        self.decoder = Decoder(
            d_hidden=dec_hidden, conv_chans=ed_conv_chans, kernel_size=ed_kernel_size, n_heads=dec_heads, n_layers=dec_layers, 
            dropout=dec_dropout, eps=eps, max_seq_len=max_seq_len
        )
        
        self.postnet = PostNet(
            n_mel_chans=n_mel_chans, conv_chans=pn_conv_chans, kernel_size=pn_kernel_size, n_layers=pn_layers,
            act_fn=pn_act_fn, dropout=pn_dropout
        )
    
    
    def forward(self, x: Tensor, src_mask: Tensor, mel_mask: Tensor):
        #TODO: To be finished tmr
        pass
    
    
def get_fastspeech2(config) -> FastSpeech2:
    #TODO: Initialize after finalize config file structure
    model = FastSpeech2()
    for param in model.parameters():
        if param.dim() > 1:
            init.xavier_uniform_(param)
    return model

    