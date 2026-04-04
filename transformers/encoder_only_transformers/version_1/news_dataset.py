import torch 
import torch.nn as nn 
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, ds, tokenizer, max_seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        example = self.ds[idx]
        encoding = self.tokenizer(text=example["text"], 
                                  padding="max_length", 
                                  truncation=True, 
                                  max_length=self.max_seq_len, 
                                  return_tensors="pt")        
        return {
            "input_ids": encoding["input_ids"].squeeze(), #(seq_len)
            "mask": encoding["attention_mask"].unsqueeze(0).int(), #(1,1,seq_len)
            "label": torch.tensor(example["label"], dtype=torch.int64), 
        }
