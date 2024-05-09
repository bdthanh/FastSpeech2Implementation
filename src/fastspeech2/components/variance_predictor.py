from torch import Tensor
from torch.nn import Module, Conv1d, ReLU, Dropout, Linear
from .layer_normalization import LayerNorm

class VariancePredictor(Module):
    #TODO: Need to polish this class, the description in the paper is ambiguous
    def __init__(self, d_input, d_hidden: int = 256, kernel_size: int = 3, filter_size: int = 256, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1d_1 = Conv1d()
        self.relu_1 = ReLU()
        self.layer_norm_1 = LayerNorm()
        self.dropout_1 = Dropout(p=dropout)
        self.conv1d_2 = Conv1d()
        self.relu_2 = ReLU()
        self.layer_norm_2 = LayerNorm()
        self.dropout_2 = Dropout(p=dropout)
        self.linear = Linear()
        
        
    def forward(self, x: Tensor, mask: Tensor):
        x = self.relu_1(self.conv1d_1(x))
        x = self.dropout_1(self.layer_norm_1(x))
        x = self.relu_2(self.conv1d_2(x))
        x = self.dropout_2(self.layer_norm_2(x))
        x = self.linear(x)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
            
        return x
        