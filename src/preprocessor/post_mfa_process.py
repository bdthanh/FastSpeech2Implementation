import os
import random
import json
import argparse
import yaml
import tgt
import librosa
import numpy as np
import pyworld as pw
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from audio.stft import TacotronSTFT
from audio.audio_processing import get_mel_from_wav

class Preprocessor:
    
    def __init__(self, config) -> None:
        self.config = config
        self.project_dir = config['path']['project_path']
        self.pre_mfa_dir = os.path.join(self.project_dir, config['path']['pre_mfa_path'])
        self.speaker = config['vocoder']['speaker']
        self.post_mfa_dir = os.path.join(self.project_dir, config['path']['post_mfa_path'], self.speaker)
        self.out_dir = os.path.join(self.project_dir, config['path']['final_path'])
        self.sampling_rate = config['preprocessing']['audio']['sampling_rate']
        self.max_wav_value = config['preprocessing']['audio']['max_wav_value']
        self.hop_length = config['preprocessing']['stft']['hop_length']
        self.pitch_norm = config['preprocessing']['pitch']['normalization']
        self.energy_norm = config['preprocessing']['energy']['normalization']
        self.val_size = config['preprocessing']['val_size']
        self.STFT = TacotronSTFT(
            config['preprocessing']['stft']['filter_length'],
            config['preprocessing']['stft']['hop_length'],
            config['preprocessing']['stft']['win_length'],
            config['preprocessing']['mel']['n_mel_channels'],
            config['preprocessing']['audio']['sampling_rate'],
            config['preprocessing']['mel']['mel_fmin'],
            config['preprocessing']['mel']['mel_fmax'],
        )
    
    def process(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, 'mel')), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, 'pitch')), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, 'energy')), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, 'duration')), exist_ok=True)
        
        print('------------------START PROCESSING------------------')
        out, n_frames = [], 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        for wav_file in tqdm(os.listdir(self.pre_mfa_dir), total=13100, desc="Processing audio & textgrid"):
            if ".wav" not in wav_file:
                continue
            basename = wav_file.split('.')[0]
            textgrid_path = os.path.join(
                self.out_dir, "TextGrid", self.speaker, "{}.TextGrid".format(basename)
            )
            if os.path.exists(textgrid_path):
                outcome = self.process_utterance(basename)
                if outcome is None:
                    continue
                info, pitch, energy, durs = outcome
                out.append(info)
            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))

            n_frames += durs
            
        pitch_mean = pitch_scaler.mean_[0] if self.pitch_norm else 0
        pitch_std = pitch_scaler.scale_[0] if self.pitch_norm else 1
        energy_mean = energy_scaler.mean_[0] if self.energy_norm else 0 
        energy_std = energy_scaler.scale_[0] if self.energy_norm else 1
        
        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )
                
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))
            
        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out
        
    def process_utterance(self, basename):
        wav_path = os.path.join(self.pre_mfa_dir, self.speaker, f'{basename}.wav')
        txt_path = os.path.join(self.pre_mfa_dir, self.speaker, f'{basename}.lab')
        textgrid_path = os.path.join(
            self.post_mfa_dir, "TextGrid", self.speaker, "{}.TextGrid".format(basename)
        )
        textgrid = tgt.io.read_textgrid(textgrid_path)
        phones, durs, start_time, end_time = self.get_alignment(textgrid.get_tier_by_name("phones"))
        phones_str = '{' + ' '.join(phones) + '}'
        if start_time >= end_time:
            return None 
        wav, _ = librosa.load(wav_path)
        wav = wav[int(self.sampling_rate * start_time):int(self.sampling_rate * end_time)].astype(np.float32)
        with open(txt_path, 'r') as f:
            raw_text = f.readline().strip("\n")
        pitch, t = pw.dio(wav.astype(np.float64), self.sampling_rate, frame_preriod=self.hop_length/self.sampling_rate)
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[:sum(durs)]
        if np.sum(pitch != 0) <= 1:
            return None
        mel_spectrogram, energy = get_mel_from_wav(wav, self.STFT)        
        mel_spectrogram = mel_spectrogram[:, : sum(durs)]
        energy = energy[: sum(durs)]
        
        dur_filename = "{}-duration-{}.npy".format(self.speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), durs)

        pitch_filename = "{}-pitch-{}.npy".format(self.speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(self.speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(self.speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )
        return (
            "|".join([basename, self.speaker, phones_str, raw_text]),
            #TODO: See if removing outlier is needed
            # self.remove_outlier(pitch),
            # self.remove_outlier(energy),
            pitch, energy,
            mel_spectrogram.shape[1]
        )

     
    def get_alignment(self, tier):
        sil_phones = set(["sil", "sp", "spn"]) #TODO: Check if "" is a silence phone
        phones, durs = [], []
        start_time, end_time, end_id = 0, 0, 0
        for t in tier._objects:
            start, end, phone = t.start_time, t.end_time, t.text
            if len(phones) == 0:
                if phone not in sil_phones:
                    start_time = start
                    
            if phone not in sil_phones:
                end_time = end
                end_id = len(phones)
            phones.append(phone)
            durs.append(int(
                np.round(end * self.sampling_rate / self.hop_length) - 
                np.round(start * self.sampling_rate / self.hop_length)
            ))
        phones = phones[:end_id]
        durs = durs[:end_id]
        
        return phones, durs, start_time, end_time
            
                
    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]


    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help='path to preprocess.yaml', 
        default='C:/Users/e0817820/Desktop/Project/FastSpeech2Implementation/config/config.yaml'
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)    
    preprocessor = Preprocessor(config)
    preprocessor.process()
        
            