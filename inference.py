import re
import torch
import argparse
import numpy as np
from g2p_en import G2p
from string import punctuation
from src.utils import load_config, is_file_exist, choose_device, get_mask_from_lengths
from src.dataset.symbol_vocabulary import SymbolVocabulary
from src.model.fastspeech2 import get_fastspeech2
from src.vocoder.models import get_vocoder
from pathlib import Path

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ").replace("{", "").replace("}", "")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(SymbolVocabulary().symbols_to_ids(phones.split(" ")))

    return np.array(sequence)

def load_checkpoint_if_exists(path, model):
    if is_file_exist(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def inference(model, device, config, vocoder, phonemes, phonemes_len, max_phoneme_len, p_control, e_control, d_control):
    src_masks = get_mask_from_lengths(phonemes_len, device, max_phoneme_len)
    output = model(phonemes, src_masks)
    mel_pred = output[0]
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--e_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--d_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()
    
    
    args = parser.parse_args()
    current_file_path = Path(__file__).resolve() 
    current_dir = current_file_path.parent 
    config_path = current_dir / 'config' / 'config.yaml'
    config = load_config(config_path)
    device = choose_device()
    model = get_fastspeech2(config, len(SymbolVocabulary()), device)
    model = load_checkpoint_if_exists(config['path']['checkpoint_last'], model)
    vocoder = get_vocoder(config, device)
    
    while text := input("Enter your script (less than 30 words):"):
        phonemes = np.array([preprocess_english(text, config)])
        phonemes_len = np.array([len(phonemes[0])])
        max_phoneme_len = phonemes_len
        item = (phonemes, phonemes_len, max_phoneme_len)
        inference(model, config, vocoder, torch.Tensor(phonemes).to(device), phonemes_len, max_phoneme_len, args.p_control, args.e_control, args.d_control)
        
    
    