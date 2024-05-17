import os
import json
import torch
import numpy as np 
from src.utils import pad_1D, pad_2D

from torch.utils.data import Dataset

class ParallelDataset(Dataset):
    
    def __init__(self, filename, config):
        self.dataset_name = config["dataset"]
        self.project_path = config['path']['project_path']
        self.post_mfa_path = config["path"]["post_mfa_path"]
        self.final_path = config['path']['final_path']
        self.cleaners = config["preprocessing"]["text"]["text_cleaners"]

        self.basenames, self.speakers, self.phonemes, self.raw_texts = self.process_metadata(
            filename
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basenames[idx]
        speaker = self.speakers[idx]
        raw_text = self.raw_texts[idx]
        #TODO: Implement mapping phoneme to id
        phoneme_enc = np.array(phoneme_to_id(self.phonemes[idx], self.cleaners))
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
                basenames.append(basename)
                speakers.append(speaker)
                phonemes.append(phoneme)
                raw_texts.append(raw_text)
            return basenames, speakers, phonemes, raw_texts

    def reprocess(self, data, idxs):
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
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            phonemes_lens,
            max(phonemes_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )
        
    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
    
def get_target_path(speaker, component, basename):
    return f"{speaker}-{component}-{basename}.npy"