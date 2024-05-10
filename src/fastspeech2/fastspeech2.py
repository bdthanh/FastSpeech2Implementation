from torch.nn import Module
from .encoder import Encoder
from .variance_adaptor import VarianceAdaptor
from .decoder import Decoder
from .postnet import PostNet

class FastSpeech2(Module):
    #TODO: To be implemented   
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()
        self.postnet = PostNet()
    
    
    def forward(self, x):
        pass