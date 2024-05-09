from torch.nn import Module, Embedding

class PhonemeEmbedding(Module):
    
    def __init__(self, phoneme_size: int, d_hidden: int = 256) -> None:
        super().__init__()
        self.embedding = Embedding(phoneme_size, d_hidden, padding_idx=0)
        
        
    def forward(self, x):
        return self.embedding(x)
        