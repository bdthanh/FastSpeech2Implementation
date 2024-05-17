"Modified from https://github.com/ming024/FastSpeech2/blob/master/prepare_align.py"

import os 
import yaml
import argparse
import librosa
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from text import _clean_text

def pre_mfa_process(config):
    project_dir = config['path']['project_path']
    input_dir = os.path.join(project_dir, config['path']['original_path'])
    metadata_csv = os.path.join(input_dir, 'metadata.csv')
    output_dir = os.path.join(project_dir, config['path']['pre_mfa_path'])
    sampling_rate = config['preprocessing']['audio']['sampling_rate']
    max_wav_value = config['preprocessing']['audio']['max_wav_value']
    dataset = 'LJSpeech'
    with open(metadata_csv, encoding='utf-8') as f:
        for line in tqdm(f, total=13100, desc='Pre MFA Processing'):
            parts = line.strip().split('|')
            base_name = parts[0]
            text = parts[2]
            text = _clean_text(text, ['english_cleaners'])
            
            wav_path = os.path.join(input_dir, 'wavs', f'{base_name}.wav')
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
                wav, _ = librosa.load(wav_path, sr=sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(output_dir, dataset, '{}.wav'.format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(output_dir, dataset, '{}.lab'.format(base_name)),
                    'w',
                ) as f1:
                    f1.write(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help='path to preprocess.yaml', 
        default='C:/Users/e0817820/Desktop/Project/FastSpeech2Implementation/config/config.yaml'
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    pre_mfa_process(config=config)
    