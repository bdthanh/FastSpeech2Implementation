import os
import torch
import yaml
import numpy as np 
from utils import pad_1D, pad_2D

from torch.utils.data import Dataset, DataLoader
from symbol_vocabulary import SymbolVocabulary

class ParallelDataset(Dataset):
    
    def __init__(self, symbol_vocab: SymbolVocabulary, filename, config):
        self.dataset_name = config["dataset"]
        self.project_path = config['path']['project_path']
        self.post_mfa_path = config["path"]["post_mfa_path"]
        self.final_path = config['path']['final_path']
        self.cleaners = config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = config["optimizer"]["batch_size"]
        self.symbol_vocab = symbol_vocab
        self.drop_last = False
        self.basenames, self.speakers, self.phonemes, self.raw_texts = self.process_metadata(
            filename
        )

    def __len__(self):
        return len(self.phonemes)

    def __getitem__(self, idx):
        basename = self.basenames[idx]
        speaker = self.speakers[idx]
        raw_text = self.raw_texts[idx]
        phoneme_enc = np.array(self.symbol_vocab.symbols_to_ids(self.phonemes[idx]))
        mel_path = os.path.join(self.final_path, "mel", get_target_path(speaker, "mel", basename))
        mel = np.load(mel_path)
        pitch_path = os.path.join(self.final_path, "pitch", get_target_path(speaker, "pitch", basename))
        pitch = np.load(pitch_path)
        energy_path = os.path.join(self.final_path, "energy", get_target_path(speaker, "energy", basename))
        energy = np.load(energy_path)
        duration_path = os.path.join(self.final_path, "duration", get_target_path(speaker, "duration", basename))
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker,
            "phoneme": phoneme_enc,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_metadata(self, metadata_filename):
        with open(os.path.join(self.final_path, metadata_filename), "r", encoding="utf-8") as f:
            basenames = []
            speakers = []
            phonemes = []
            raw_texts = []
            for line in f.readlines():
                basename, speaker, phoneme, raw_text = line.strip("\n").split("|")
                phoneme = phoneme.rstrip("}").lstrip("{").split(" ")
                basenames.append(basename)
                speakers.append(speaker)
                phonemes.append(phoneme)
                raw_texts.append(raw_text)
            return basenames, speakers, phonemes, raw_texts

    def process_for_batch(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        phonemes = [data[idx]["phoneme"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        phonemes_lens = np.array([phoneme.shape[0] for phoneme in phonemes])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        phonemes = pad_1D(phonemes)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return {
            'ids': ids,
            'raw_texts': raw_texts,
            'speakers': speakers,
            'phonemes': phonemes,
            'phonemes_lens': phonemes_lens,
            'max_phoneme_len': max(phonemes_lens),
            'mels': mels,
            'mel_lens': mel_lens,
            'max_mel_len': max(mel_lens),
            'pitches': pitches,
            'energies': energies,
            'durations': durations
        }
        
    def collate_fn(self, data):
        data_size = len(data)
        idx_arr = np.arange(data_size).tolist()
        return self.process_for_batch(data, idx_arr)
    
def get_target_path(speaker, component, basename):
    return f"{speaker}-{component}-{basename}.npy"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.load(
        open(r"C:\Users\e0817820\Desktop\Project\FastSpeech2Implementation\config\config.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = ParallelDataset(
        SymbolVocabulary(), "train.txt", config
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    i=0
    for batch in train_loader:
        i+=1
        print(i)