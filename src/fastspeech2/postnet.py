import torch
from torch import Tensor
from copy import deepcopy
from torch.nn import Module, Conv1d, BatchNorm1d, Tanh, ReLU, ModuleList, Dropout

#This PostNet arch is adopted from Tacotron 2 paper
class PostNet(Module):
    
    def __init__(self, n_mel_chans, filter_size, kernel_size, n_layers = 5, act_fn = "tanh", dropout = 0.2) -> None:
        super().__init__()
        first_layer = PostNetLayer(n_mel_chans, filter_size, kernel_size, act_fn, dropout)
        single_middle_layer = PostNetLayer(filter_size, filter_size, kernel_size, act_fn, dropout)
        last_layer = PostNetLayer(filter_size, n_mel_chans, kernel_size, None, dropout)
        self.postnet_layers = ModuleList([first_layer] + [deepcopy(single_middle_layer) for _ in range(n_layers)] + [last_layer])
        
        
    def forward(self, x: Tensor):
        x = x.contiguous().transpose(1, 2)
        for layer in self.postnet_layers:
            x = layer(x)
        x = x.contiguous().transpose(1, 2)    
            
        return x
    
    
class PostNetLayer(Module):
    #TODO: To implement BatchNorm1d if possible
    #TODO: Try ReLU instead to see which one is better ReLU or Tanh (Tanh is from the paper)
    def __init__(self, in_chans = 80, out_chans = 512, kernel_size = 5, activation_fn = None, dropout = 0.2) -> None:
        super().__init__()
        self.conv1d = Conv1d(in_chans, out_chans, kernel_size, padding=int(kernel_size-1) // 2)
        self.batchnorm1d = BatchNorm1d(out_chans, eps=1e-9)
        if activation_fn == "tanh":
            self.act_fn = Tanh()
        elif activation_fn == "relu":
            self.act_fn = ReLU()
        else:
            self.act_fn = None
        self.dropout = Dropout(p=dropout)
        
        
    def forward(self, x):
        x = self.batchnorm1d(self.conv1d(x))
        if self.act_fn != None: 
            x = self.act_fn(x)
        
        return self.dropout(x)
        