import os
import torch

from pathlib import Path
from transformers import  GPT2Tokenizer


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 600000,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "vocab_size": 50304,
        "datasource": 'shakespeare',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel"
    }

current_directory = os.path.dirname(os.path.abspath(__file__))

def get_weights_file_path(config, epoch: str):
    model_folder = f"{current_directory}/{config['datasource']}/{config['model_folder']}"
    # Create the folder and subfolders if they don't exist
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return model_folder + '/' + model_filename

def get_data_folder_path(config):
    model_folder = f"{current_directory}/{config['datasource']}/data"
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    return model_folder 

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{current_directory}/{config['datasource']}/{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def get_gpt2_tokenizer(config):
    tokenizer:GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        model_max_length=config['seq_len'],
        pad_token='[PAD]')
    return tokenizer

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask==0