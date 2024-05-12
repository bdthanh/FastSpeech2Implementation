import os
import json
import yaml
import torch 

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