from torch import Tensor
from torch.nn import Module, Conv1d, ReLU, Dropout

class PositionWiseFeedForward(Module):
    
    def __init__(self, d_in, d_hidden, kernel_size, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1d_1 = Conv1d(
            d_in, d_hidden, kernel_size=kernel_size, padding=(kernel_size-1)//2
        )
        self.relu_1 = ReLU()
        self.conv1d_2 = Conv1d(
            d_hidden, d_in, kernel_size=1, padding=(1 - 1) // 2
        )
        self.relu_2 = ReLU()
        self.dropout = Dropout(p=dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.conv1d_1(x)
        x = self.relu_1(x)
        x = self.conv1d_2(x)
        
        return self.dropout(x.transpose(1, 2))
