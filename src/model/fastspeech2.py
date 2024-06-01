from torch.nn import Module, init
from .encoder import Encoder
from .variance_adaptor import VarianceAdaptor
from .decoder import Decoder
from .postnet import PostNet
from .components.positional_encoding import get_positional_encoding

class FastSpeech2(Module):
    
    def __init__(self, phoneme_size, max_seq_len, enc_hidden, enc_heads, enc_layers, dec_hidden, dec_heads, 
                 dec_layers, ed_conv_chans, ed_kernel_size, enc_dropout, dec_dropout, var_conv_chans, var_kernel_size, 
                 n_bins, var_dropout, n_mel_chans, pn_conv_chans, pn_kernel_size, pn_layers, 
                 pn_dropout, pn_act_fn, eps) -> None:
        super().__init__()
        self.encoder = Encoder(
            phoneme_size=phoneme_size, d_hidden=enc_hidden, conv_chans=ed_conv_chans, kernel_size=ed_kernel_size,
            n_heads=enc_heads, n_layers=enc_layers, dropout=enc_dropout, eps=eps, max_seq_len=max_seq_len
        )
        
        self.variance_adaptor = VarianceAdaptor(
            d_in=enc_hidden, conv_chans=var_conv_chans, kernel_size=var_kernel_size, dropout=var_dropout, n_bins = n_bins
        )
        
        self.decoder = Decoder(
            d_hidden=dec_hidden, conv_chans=ed_conv_chans, n_mel_chans=n_mel_chans, kernel_size=ed_kernel_size, 
            n_heads=dec_heads, n_layers=dec_layers, dropout=dec_dropout, eps=eps, max_seq_len=max_seq_len
        )
        
        self.postnet = PostNet(
            n_mel_chans=n_mel_chans, conv_chans=pn_conv_chans, kernel_size=pn_kernel_size, n_layers=pn_layers,
            act_fn=pn_act_fn, dropout=pn_dropout
        )
    
    
    def forward(self, x, src_mask, mel_mask, dur_trg, pitch_trg, energy_trg, max_dur=None, p_control=1.0, e_control=1.0, d_control=1.0):
        x = self.encoder(x, src_mask)
        x, mel_mask, log_dur_pred, dur_rounded, pitch_pred, pitch_emb, energy_pred, energy_emb = self.variance_adaptor(
            x, dur_trg, pitch_trg, energy_trg, src_mask, mel_mask, max_dur, p_control, e_control, d_control    
        )
        x = self.decoder(x, mel_mask)
        postnet_x = self.postnet(x)
        
        return (x, mel_mask, postnet_x, log_dur_pred, dur_rounded, pitch_pred, pitch_emb, energy_pred, energy_emb)
    
    
def get_fastspeech2(config, phoneme_size, device) -> FastSpeech2:
    max_seq_len = config["max_seq_len"]
    model_config = config["fastspeech2"]
    enc_hidden = model_config["enc_hidden"]
    enc_heads = model_config["enc_heads"]
    enc_layers = model_config["enc_layers"]
    dec_hidden = model_config["dec_hidden"]
    dec_heads = model_config["dec_heads"]
    dec_layers = model_config["dec_layers"]
    ed_conv_chans = model_config["conv_filter_size"]
    ed_kernel_size = model_config["conv_kernel_size"]
    enc_dropout = model_config["enc_dropout"]
    dec_dropout = model_config["dec_dropout"]
    var_conv_chans = model_config["var_conv_filter_size"]
    var_kernel_size = model_config["var_conv_kernel_size"]
    n_bins = config["variance_embedding"]["n_bins"]
    var_dropout = model_config["var_dropout"]
    n_mel_chans = config["preprocessing"]["mel"]["n_mel_channels"]
    pn_conv_chans = model_config["postnet_filter_size"]
    pn_kernel_size = model_config["postnet_kernel_size"]
    pn_layers = model_config["postnet_layers"]
    pn_dropout = model_config["postnet_dropout"] 
    pn_act_fn = model_config["postnet_act_fn"]
    eps = config["optimizer"]["eps"]
    
    model = FastSpeech2(
        phoneme_size, max_seq_len, enc_hidden, enc_heads, enc_layers, dec_hidden, dec_heads, dec_layers, 
        ed_conv_chans, ed_kernel_size, enc_dropout, dec_dropout, var_conv_chans, var_kernel_size, n_bins, 
        var_dropout, n_mel_chans, pn_conv_chans, pn_kernel_size, pn_layers, pn_dropout, pn_act_fn, eps
    ).to(device)
    
    for param in model.parameters():
        if param.dim() > 1:
            init.xavier_uniform_(param)
    return model

    