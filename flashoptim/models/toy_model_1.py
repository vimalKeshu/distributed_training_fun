import torch
import torch.nn as nn

class ToyTransformerLM(nn.Module):
    def __init__(self, vocab_size=50257, d_model=256, nhead=4,
                 num_layers=2, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=input_ids.device)
        # Match mask dtype to weight dtype (critical when model is bf16)
        weight_dtype = self.tok_emb.weight.dtype
        if weight_dtype != mask.dtype:
            mask = mask.to(dtype=weight_dtype)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)