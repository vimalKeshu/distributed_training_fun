import torch
import torch.nn as nn 
import math


class WordEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):       
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        
        Returns:
            (batch_size, seq_len, d_model), with word embeddings.
        """        

        # multiply by sqrt(d_model) to scale the embeddings 
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            (batch_size, seq_len, d_model), with position embeddings.
        """        
        batch_size, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        return x + self.dropout(self.position_embedding(positions))
    
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Layer Normalization as used in Transformer models.

        Args:
            features (int): Number of features (hidden size of model).
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable bias

    def forward(self, x):
        """
        Applies Layer Normalization.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tensor: Normalized tensor of shape (batch, seq_len, hidden_size).
        """
        mean = x.mean(dim=-1, keepdim=True)  # Compute mean
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # Compute variance
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Feed Forward Network used in Transformer models.

        Args:
            d_model (int): Model's hidden size.
            d_ff (int): Hidden size of feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # First transformation
        self.activation = nn.GELU()  # GELU activation (better than ReLU for Transformers)
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.linear_2 = nn.Linear(d_ff, d_model)  # Second transformation

    def forward(self, x):
        """
        Forward pass for Feed Forward Network.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))

class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float):
        """
        Residual Connection with Layer Normalization (Post-Norm)

        Args:
            features (int): Hidden size of the model.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)  # LayerNorm at the end

    def forward(self, x, sublayer):
        """
        Forward pass for Residual Connection.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, features).
            sublayer (nn.Module): Transformer sublayer (e.g., self-attention or feedforward).

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, features).
        """
        return self.norm(x + self.dropout(sublayer(x)))  # Post-Norm


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        """
        Multi-Head Self-Attention Block for an Encoder-Only Transformer.

        Args:
            d_model (int): Hidden size of the model.
            h (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # Dimension per head

        # Linear transformations for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model, bias=True)  
        self.w_k = nn.Linear(d_model, d_model, bias=True)  
        self.w_v = nn.Linear(d_model, d_model, bias=True)  
        self.w_o = nn.Linear(d_model, d_model, bias=True)  

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        """
        Scaled Dot-Product Attention.

        Args:
            query (Tensor): (batch, h, seq_len, d_k)
            key (Tensor): (batch, h, seq_len, d_k)
            value (Tensor): (batch, h, seq_len, d_k)
            mask (Tensor or None): Mask tensor (batch, 1, 1, seq_len)
            dropout (nn.Dropout): Dropout layer

        Returns:
            Tensor: (batch, h, seq_len, d_k) - Attention output
            Tensor: (batch, h, seq_len, seq_len) - Attention scores
        """
        d_k = query.shape[-1]
        # Compute scaled dot-product attention scores
        attention_scores = torch.einsum("bhqd, bhkd -> bhqk", query, key) / math.sqrt(d_k)  

        if mask is not None:
            mask = mask.to(torch.bool)  # Ensure mask is boolean
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_scores = attention_scores.softmax(dim=-1)  # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.einsum("bhqk, bhvd -> bhqd", attention_scores, value), attention_scores

    def forward(self, x, mask=None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).
            mask (Tensor, optional): Mask tensor of shape (batch, 1, 1, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, d_model).
            Tensor: Attention scores of shape (batch, h, seq_len, seq_len).
        """
        batch_size, seq_len, _ = x.shape

        # Apply linear transformations
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        # Reshape to (batch, seq_len, h, d_k) → (batch, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        # Compute attention
        x, attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Reshape back to (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(x), attention_scores


class EncoderBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, 
                 dropout: float) -> None:
        """
        Transformer Encoder Block.

        Args:
            d_model (int): Hidden dimension size.
            self_attention_block (MultiHeadAttentionBlock): Self-attention layer.
            feed_forward_block (FeedForwardBlock): Position-wise feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, mask):
        """
        Forward pass of the Transformer Encoder Block.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).
            mask (Tensor): Mask tensor of shape (batch, 1, 1, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention_block(x, mask)  # Get only the output
        x = self.residual_connections[0](x, lambda x: attn_output)

        # Feed-forward with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        """
        Transformer Encoder with multiple stacked encoder blocks.

        Args:
            d_model (int): The hidden dimension size of the model.
            layers (nn.ModuleList): A list of EncoderBlock instances.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)  # Final normalization (like BERT)

    def forward(self, x, mask):
        """
        Forward pass through multiple encoder blocks.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor): Mask tensor of shape (batch_size, 1, 1, seq_len).

        Returns:
            Tensor: The encoded representation (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)  # Pass through each encoder block
        return self.norm(x)  # Apply final normalization


class ClassificationLayer(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        """
        Classification layer for Transformer Encoder.

        Args:
            d_model (int): Embedding dimension.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),  # Optional non-linearity
            nn.ReLU(),  # Activation for better learning
            nn.Dropout(dropout),  # Prevent overfitting
            nn.Linear(d_model, num_classes)  # Final classifier
        )

    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x (Tensor): Input tensor of shape (batch_size, d_model).
        
        Returns:
            Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.classifier(x)

class EncoderOnlyTransformer(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, 
                 num_layers: int, num_classes: int, max_seq_len: int, dropout: float = 0.1):
        """
        Encoder-Only Transformer Model.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of feedforward layer.
            num_layers (int): Number of encoder blocks.
            num_classes (int): Number of output classes.
            max_seq_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # Embedding Layers
        self.word_embedding = WordEmbeddings(d_model, vocab_size)
        self.position_embedding = PositionalEmbeddings(d_model, max_seq_len, dropout)

        # Transformer Encoder Layers
        encoder_layers = nn.ModuleList([
            EncoderBlock(
                d_model, 
                MultiHeadAttentionBlock(d_model, num_heads, dropout), 
                FeedForwardBlock(d_model, d_ff, dropout), 
                dropout
            ) for _ in range(num_layers)
        ])
        self.encoder = Encoder(d_model, encoder_layers)

        # Classification Layer
        self.classification_layer = ClassificationLayer(d_model, num_classes, dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through the Transformer Encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len).
            mask (Tensor, optional): Attention mask of shape (batch_size, 1, 1, seq_len).
        
        Returns:
            Tensor: Logits of shape (batch_size, num_classes).
        """
        # Embedding and Positional Encoding
        x = self.word_embedding(x) 
        x = self.position_embedding(x)

        # Pass through encoder layers
        x = self.encoder(x, mask)

        # Classification (Use only [CLS] token representation)
        # In models like BERT, the first token in the input sequence is typically a special classification token ([CLS]).
        # The Transformer learns to store global sentence-level information in the [CLS] token during training.
        # This token's representation is then used for downstream tasks like classification.    
        # Alternative: Some models use mean pooling over all token embeddings instead:
        # x = x.mean(dim=1)  # Average over all tokens    
        x = x[:, 0, :]  # Extract [CLS] token representation
        # Map to output classes
        logits = self.classification_layer(x)  

        return logits

