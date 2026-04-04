from shakespeare_config import (get_config, 
                                latest_weights_file_path, 
                                get_gpt2_tokenizer,
                                causal_mask,
                                current_directory)
import torch 
import warnings
import heapq
from train import build_transformer

def predict_with_greedy_search(start_str:str)-> None:
    config:dict=get_config() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = get_gpt2_tokenizer(config=config)
    model = build_transformer(vocab_size=config['vocab_size'],
                              seq_len=config['seq_len'],
                              d_model=config['d_model']).to(device)
    # load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    output = start_str
    with torch.no_grad():
        start_tokens = tokenizer.encode(start_str)
        print(start_tokens)
        input = torch.tensor(data=start_tokens, dtype=torch.int64).unsqueeze(dim=0).to(device)
        # print(input)
        while input.size(1) <= config['seq_len']:
            # use mask otheriwse model may generate repetitive words in prediction
            mask = causal_mask(input.size(1)).to(device)
            out = model.decode(input,mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            input = torch.cat(
                    [
                        input, 
                        torch.empty(1,1).type_as(input).fill_(next_word.item()).to(device)
                    ],
                    dim=1
                )
            output += tokenizer.decode(next_word.item())
    
    print(f'Model output: {output}')


def predict_with_beam_search(start_str: str, 
                             beam_width: int = 3) -> None:
    config: dict = get_config() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = get_gpt2_tokenizer(config=config)
    model = build_transformer(vocab_size=config['vocab_size'],
                              seq_len=config['seq_len'],
                              d_model=config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # Initial input
    start_tokens = tokenizer.encode(start_str)
    input = torch.tensor(data=start_tokens, dtype=torch.int64).unsqueeze(dim=0).to(device) # (1, seq_len)

    # Beam search variables
    beams = [(0, input, [])]  # Each beam is a tuple of (score, sequence, tokens_generated)

    for _ in range(config['seq_len']):
        all_candidates = []

        # Process each beam
        for score, seq, tokens in beams:
            # use mask otheriwse model may generate repetitive words in prediction
            mask = causal_mask(seq.size(1)).to(device)
            out = model.decode(seq, mask)
            prob = model.project(out[:, -1])

            # Get the top k predictions
            top_k_probabilities, top_k_indices = torch.topk(prob, beam_width, dim=1)

            # Generate new beams for each of the top k tokens
            for i in range(beam_width):
                new_token = top_k_indices[0, i].item()
                new_score = score - torch.log(top_k_probabilities[0, i]).item()  # We negate because we want to maximize
                new_seq = torch.cat([seq, torch.tensor([[new_token]], device=device)], dim=1)
                new_tokens = tokens + [new_token]
                all_candidates.append((new_score, new_seq, new_tokens))

        # Sort all candidates based on their score and keep the top `beam_width` beams
        beams = heapq.nsmallest(beam_width, all_candidates, key=lambda x: x[0])

        # Optionally, stop early if all beams end with an EOS token
        if all(beam[1].shape[1] >= config['seq_len'] for beam in beams):
            break

    # Retrieve the best beam (with the highest score)
    best_beam = beams[0]
    best_tokens = best_beam[2]

    # Decode the final sequence
    output = tokenizer.decode(best_tokens, skip_special_tokens=True)
    print(f'Model output: {output}')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    start_str = 'Now sadder, that you come so'
    predict_with_greedy_search(start_str=start_str)
    print('--'*100)
    predict_with_beam_search(start_str=start_str)
