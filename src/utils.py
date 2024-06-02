import os
import json
import yaml
import torch 
import numpy as np

def choose_device():
    """Choose the device to run the model on"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(path):
    """Loads a configuration file in yaml format"""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_json(path):
    """Loads a json file"""
    with open(path, "r") as f:
        return json.load(f)

def create_if_missing_folder(path: str):
    """Creates a folder if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def is_file_exist(path: str):
    return os.path.exists(path)

def get_num_params(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def get_mask_from_lengths(lengths, device, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len).to(device) # True (1) for non-pad item, False (0) otherwise

    return mask