import torch
import numpy as np 
from typing import List

_pad = "_"
# _punctuation = "!'(),.:;? "
# _special = "-"
# _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["sp", "spn", "sil"]

_valid_phonemes = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]


# Export all symbols:
symbols = (
    [_pad]
    # + list(_special)
    # + list(_punctuation)
    # + list(_letters)
    + _valid_phonemes
    + _silences
)

class SymbolVocabulary:
    def __init__(self) -> None:
        self.symbols = symbols
        self.pad_id = 0
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def __len__(self):
        return len(self.symbol_to_id)
    
    def symbols_to_ids(self, symbols_list: List[str]):
        return torch.tensor(list(map(lambda symbol: self.symbol_to_id[symbol], symbols_list)), dtype=torch.int64)
    