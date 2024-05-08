from torch.nn import Module, Embedding

class PhonemeEmbedding(Module):
    
    def __init__(self, phoneme_size, d_enc_hidden) -> None:
        super().__init__()
        self.embedding = Embedding(phoneme_size, d_enc_hidden, padding_idx=0)
        
    def forward(self, x):
        return self.embedding(x)
        