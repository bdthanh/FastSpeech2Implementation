from torch import Tensor
from torch.nn import Module, Conv1d, ReLU, Dropout, Linear
from .layer_normalization import LayerNorm

class VariancePredictor(Module):
    
    def __init__(self, d_in: int, conv_chans: int = 256, kernel_size: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1d_1 = Conv1d(d_in, conv_chans, kernel_size=kernel_size, padding=((kernel_size - 1) // 2))
        self.relu_1 = ReLU()
        self.layer_norm_1 = LayerNorm(conv_chans)
        self.dropout_1 = Dropout(p=dropout)
        self.conv1d_2 = Conv1d(conv_chans, conv_chans, kernel_size=kernel_size, padding=((kernel_size - 1) // 2))
        self.relu_2 = ReLU()
        self.layer_norm_2 = LayerNorm(conv_chans)
        self.dropout_2 = Dropout(p=dropout)
        self.linear = Linear(conv_chans, 1)
        
        
    def forward(self, x: Tensor, mask: Tensor):
        x = x.transpose(1, 2)
        x = self.relu_1(self.conv1d_1(x))
        x = self.dropout_1(self.layer_norm_1(x))
        x = self.relu_2(self.conv1d_2(x))
        x = self.dropout_2(self.layer_norm_2(x))
        x = x.transpose(1, 2)
        x = self.linear(x).squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
            
        return x
        