from torch import Tensor
from torch.nn import Module, Conv1d, ReLU, Dropout, Linear
from .layer_normalization import LayerNorm

class VariancePredictor(Module):
    #TODO: shape check
    def __init__(self, d_in: int, d_hidden: int = 256, kernel_size: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1d_1 = Conv1d(d_in, d_hidden, kernel_size=kernel_size, padding=((kernel_size - 1) // 2))
        self.relu_1 = ReLU()
        self.layer_norm_1 = LayerNorm(d_hidden)
        self.dropout_1 = Dropout(p=dropout)
        self.conv1d_2 = Conv1d(d_hidden, d_hidden, kernel_size=kernel_size, padding=1)
        self.relu_2 = ReLU()
        self.layer_norm_2 = LayerNorm(d_hidden)
        self.dropout_2 = Dropout(p=dropout)
        self.linear = Linear(d_hidden, 1)
        
        
    def forward(self, x: Tensor, mask: Tensor):
        x = self.relu_1(self.conv1d_1(x))
        x = self.dropout_1(self.layer_norm_1(x))
        x = self.relu_2(self.conv1d_2(x))
        x = self.dropout_2(self.layer_norm_2(x))
        x = self.linear(x).squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
            
        return x
        