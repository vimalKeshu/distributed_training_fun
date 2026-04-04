'''Ref: https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py
'''
import os
import requests
import numpy as np
from transformers import  GPT2Tokenizer
from shakespeare_config import get_data_folder_path, get_config, get_gpt2_tokenizer
from pathlib import Path

if __name__=='__main__':
    config=get_config()
    data_folder_path = get_data_folder_path(config=config)
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(data_folder_path, 'input.txt')
    tokenizer:GPT2Tokenizer = get_gpt2_tokenizer(config=config)

    print(tokenizer.model_max_length)

    if not Path(input_file_path).exists():
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

        data=''
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.rstrip())>0:
                    data += ' ' + line    
        
        print(data)
        n = len(data)
        train_split = int(n*0.9)
        train_data = data[:train_split]
        test_data = data[train_split:]    

        train_ids = tokenizer.encode(train_data)
        test_ids = tokenizer.encode(test_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"test has {len(test_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        test_ids = np.array(test_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(data_folder_path, 'train.bin'))
        test_ids.tofile(os.path.join(data_folder_path, 'test.bin'))
        # train has 292,080 tokens
        # test has 34,806 tokens
        print(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id))
        print(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id))