import torchmetrics.classification
from torchmetrics.text import ROUGEScore
from model import build_transformer
from shakespeare_config import (get_config, 
                                get_data_folder_path, 
                                get_weights_file_path, 
                                latest_weights_file_path,
                                current_directory,
                                causal_mask,
                                get_gpt2_tokenizer)

import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import numpy as np
import warnings
import os
from pathlib import Path

def get_model(config):
    model = build_transformer(vocab_size=config['vocab_size'],
                              seq_len=config['seq_len'],
                              d_model=config['d_model'])
    return model

def get_batch(split, data_dir, block_size, batch_size, device='gpu', device_type='cuda'):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # if device_type == 'cuda':
    #     # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    # else:
    #     x, y = x.to(device), y.to(device)
    return x, y

def greedy_decode(model, 
                  input, 
                  mask,
                  tokenizer,
                  max_len,
                  device):
    while True:
        if input.size(1) == max_len:
            break

        out = model.decode(input, mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        input = torch.cat(
            [input, torch.empty(1,1).type_as(input).fill_(next_word.item()).to(device)],
            dim=1
        )
        if next_word == tokenizer.eos_token_id:
            break 
    return input.squeeze(0)

def run_validation(model,
                   x,
                   y,
                   tokenizer,
                   max_len,
                   device,
                   print_msg,
                   global_step,
                   writer,
                   rouge:ROUGEScore):
    
    model.eval()
    source_texts = []
    expected = []
    predicted = []
    
    with torch.no_grad():
        decoder_input = x.to(device) # (b, seq)
        mask = causal_mask(x.size(1)).to(device) #(b,1,1,seq)

        # check that batch size is 1
        assert decoder_input.size(0)==1, "batch size  must be 1 for validation"

        model_out = greedy_decode(model,
                                  decoder_input, 
                                  mask, 
                                  tokenizer,
                                  max_len, 
                                  device)
        
        source_text = tokenizer.decode(x[0])
        target_text = tokenizer.decode(y[0])
        model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

        source_texts.append(source_text)
        expected.append(target_text)
        predicted.append(model_out_text)
        
        # Print the source, target and model output
        print_msg('-'*100)
        print_msg(f"{f'SOURCE: ':>12}{source_text}")
        print_msg(f"{f'TARGET: ':>12}{target_text}")
        print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

        rouge_score = rouge(predicted, expected)  
        print_msg(f"{f'ROUGE-1 Score: ':>12}{rouge_score['rouge1_fmeasure'].item()}")
        print_msg(f"{f'ROUGE-2 Score: ':>12}{rouge_score['rouge2_fmeasure'].item()}")
        print_msg(f"{f'ROUGE-L Score: ':>12}{rouge_score['rougeL_fmeasure'].item()}")      
        print_msg('-'*100)

        if writer:
            writer.add_scalar('validation ROUGE/ROUGE-1', rouge_score["rouge1_fmeasure"].item(), global_step)
            writer.add_scalar('validation ROUGE/ROUGE-2', rouge_score["rouge2_fmeasure"].item(), global_step)
            writer.add_scalar('validation ROUGE/ROUGE-L', rouge_score["rougeL_fmeasure"].item(), global_step)
            writer.add_scalar('validation ROUGE/ROUGE-L', rouge_score["rougeLsum_fmeasure"].item(), global_step)
            writer.flush() 

def train_model(config):
    # define the device 
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_ms or torch.backends.mps.is_available else "cpu"
    print("Using device:", device)

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device=device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'cpu'):
        print(f"device name: <mps>")
    else:
        print("It's cpu")

    device = torch.device(device)

    # make sure the weights folder exists 
    Path(f"{current_directory}/{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    tokenizer = get_gpt2_tokenizer(config=config)
    model = get_model(config).to(device) 
    # tensorboard
    writer = SummaryWriter(f"{current_directory}/{config['experiment_name']}")

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config['lr'],
                                 eps=1e-9)
    rouge:ROUGEScore = ROUGEScore()

    # if the user specified a model to preload before training, load it 
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = (latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None)
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1 
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids('[PAD]'), label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        #batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        X, y = get_batch(split='train',
                        data_dir=get_data_folder_path(config=config),
                        block_size=config['seq_len'],
                        batch_size=config['batch_size'])
        print(f'length of the batch: {len(X)}, type:{X.shape}')

        decoder_input = X.to(device) # (b, seq_len)
        decoder_mask = causal_mask(config['seq_len']).to(device) # (1, seq_len, seq_len)

        # run the tensors through the encoder, decoder and the projection layer
        decoder_output = model.decode(decoder_input, decoder_mask) # (b, seq, d_model)
        proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

        # compare the output with the label
        label = y.to(device) #(b, seq_len)

        # compute the loss using a simple cross entropy
        loss = loss_fn(proj_output.view(-1, config['vocab_size']), 
                        label.view(-1))
        #batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        print(f"loss: {loss.item():6.3f}")

        # log the loss 
        writer.add_scalar('train loss', loss.item(), global_step)
        writer.flush()

        # backpropagate the loss 
        loss.backward()

        # update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

        # run validation at the end of every epoch
        X_val, y_val = get_batch(split='val',
                        data_dir=get_data_folder_path(config=config),
                        block_size=config['seq_len'],
                        batch_size=1)
        run_validation(model, 
                       X_val,
                       y_val, 
                       tokenizer, 
                       config['seq_len'], 
                       device, 
                       lambda msg: print(msg), 
                       global_step, 
                       writer,
                       rouge)

        if epoch%1000==0 or epoch >= (config['num_epochs']-1):
            # save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)