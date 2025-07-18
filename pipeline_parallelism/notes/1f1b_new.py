import logging 
import os 
import time 
import sys
import argparse
import signal

import torch 
import torch.nn as nn 
import torch.distributed as dist 
import torch.nn.functional as dist 
from typing import Optional, Tuple, Any, Dict 
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamAwareAsyncCommHandler:
    """Handles async NCCL communications with CUDA stream management
        Default Stream: ........[Compute]......[Compute Next]...
        Send Stream:    ................[Send Result].........
        Recv Stream:    [Recv Data]............[Recv Next]....
                        ↑________________________↑
                        Overlapped execution!    
    """

    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank 
        self.world_size = world_size
        self.device = device 
        self.prev_rank = self.rank - 1 if self.rank > 0 else None
        self.next_rank = self.rank + 1 if self.rank < world_size - 1 else None

        # Communication tags
        self.FORWARD_TAG_BASE = 1000
        self.FORWARD_TAG_BASE = 2000 

        # Create dedicated CUDA streams for computation and communication 
        self.send_stream = torch.cuda.Stream(device=device)
        self.recv_stream = torch.cuda.Stream(device=device)
        logger.info(f"Rank {self.rank}: Created CUDA streams - "
                   f"send_stream={self.send_stream}, recv_stream={self.recv_stream}")
    
    def async_send_forward(self, send_tensor: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], torch.Tensor]:
        """Send tensor to next pipeline stage using dedicated stream"""
        if self.next_rank is not None:
            tag = self.FORWARD_TAG_BASE + micro_batch_id

            # Switch to send stream
            with torch.cuda.stream(self.send_stream):
                # Ensure tensor is contiguous in the send stream context 
                if not send_tensor.is_contiguous():
                    send_tensor = send_tensor.contiguous()
                
                # Record event in default stream to ensure computation is done 
                comp_done_event = torch.cuda.Event()
                torch.cuda.current_stream().record_event(comp_done_event)

                # Wait for computation to complete before sending
                self.send_stream.wait_event(comp_done_event)

                # Initiate async send on send stream 
                req = dist.isend(send_tensor, dst=self.next_rank, tag=tag)
            
            return req, send_tensor
        return None, send_tensor

    def async_recv_forward(self, recv_buffer: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
        """Receive tensor from previous pipeline stage using dedicated stream"""
        if self.prev_rank is not None:
            tag = self.FORWARD_TAG_BASE + micro_batch_id 

            # Switch to recv stream 
            with torch.cuda.stream(self.recv_stream):
                # Initiate async receive on recv stream
                req = dist.irecv(recv_buffer, src=self.prev_rank, tag=tag)
            
            return req, recv_buffer
        return None, None        

    def async_send_backward(self, send_tensor: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], torch.Tensor]:
        if self.prev_rank is not None:
            tag = self.BACKWARD_TAG_BASE + micro_batch_id
            
            with torch.cuda.stream(self.send_stream):
                if not send_tensor.is_contiguous():
                    send_tensor = send_tensor.contiguous()
                
                # Ensure gradients are computed before sending
                comp_done_event = torch.cuda.Event()
                torch.cuda.current_stream().record_event(comp_done_event)
                self.send_stream.wait_event(comp_done_event)
                
                req = dist.isend(send_tensor, dst=self.prev_rank, tag=tag)
                
            return req, send_tensor
        return None, send_tensor        
    
    def async_recv_backward(self, recv_buffer: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
        """Receive gradient tensor from next pipeline stage"""
        if self.next_rank is not None:
            tag = self.BACKWARD_TAG_BASE + micro_batch_id
            
            with torch.cuda.stream(self.recv_stream):
                req = dist.irecv(recv_buffer, src=self.next_rank, tag=tag)
                
            return req, recv_buffer
        return None, None        
    
    def wait_recv_ready(self, recv_req: Any, recv_buffer: torch.Tensor) -> torch.Tensor:
        """Wait for receive to complete and ensure data is ready for computation"""
        with torch.cuda.stream(self.recv_stream):
            # Wait for communication to complete
            recv_req.wait()

            # Record event when receive is done
            recv_done_event = torch.cuda.Event()
            self.recv_stream.record_event(recv_done_event)

        # Make default stream wait for receive completion
        torch.cuda.current_stream().wait_event(recv_done_event)

        return recv_buffer

class TransformerBlock(nn.Module):
    """A single transformer block with attention and MLP, supporting various masking strategies"""
    
    def __init__(self, rank: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, dtype: torch.dtype = torch.float16):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.dtype = dtype
        self.rank = rank
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            dtype=self.dtype
        )
        self.attention_dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(d_model, dtype=self.dtype)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, dtype=self.dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, dtype=self.dtype),
            nn.Dropout(dropout)
        )

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal (lower triangular) mask for autoregressive attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, 
                x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass with optional masking
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask: Attention mask of shape (seq_len, seq_len) or (batch_size * n_heads, seq_len, seq_len)
            key_padding_mask: Padding mask of shape (batch_size, seq_len)
            is_causal: Whether to apply causal masking (for autoregressive models)
        """
        _, seq_len, _ = x.shape
        
        # Pre-Norm + attention layer norm
        normed_x = self.norm1(x)

        # Create causal mask if requested
        if is_causal and attn_mask is None:
            attn_mask = self.create_causal_mask(seq_len, x.device)

        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(
            query=normed_x,
            key=normed_x, 
            value=normed_x,
            attn_mask=attn_mask,               # shape (seq_len,seq_len) or (B*heads, seq_len, seq_len)
            key_padding_mask=key_padding_mask,  # shape (B, seq_len)
            need_weights=False,  # Set to True if you want to return attention weights
            is_causal=is_causal
        )
        x = x + self.attention_dropout(attn_out)
        
        # Pre-Norm + Feed-forward with residual connection
        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x)
        x = x + ffn_out
        
        return x

class ToyModel(nn.Module):
    """Large transformer model with configurable parameters"""
    def __init__(
        self, 
        rank: int,
        vocab_size: int = 50000,
        d_model: int = 4096,
        n_layers: int = 48,
        n_heads: int = 32,
        d_ff: int = 16384,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.rank = rank
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, dtype=self.dtype)
        self.position_embedding = nn.Embedding(max_seq_len, d_model, dtype=self.dtype)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(rank, d_model, n_heads, d_ff, dropout, dtype=self.dtype)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.final_norm = nn.LayerNorm(d_model, dtype=self.dtype)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False, dtype=self.dtype)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_parameter_count(self):
        """Calculate total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)    

class PipelineStage(nn.Module):

    def __init__(self, rank: int, world_size: int, layers: nn.ModuleList):
        super().__init__()
        self.rank = rank 
        self.world_size = world_size
        self.layers = layers
    
    def forward(self, 
                x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        logger.info(f"Rank {self.rank}: forward started...., input shape: {x.shape}")

        if self.rank == 0:
            # First stage: embeddings + transformer layers
            batch_size, seq_len = x.shape
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)            
            
            # Token embeddings + position embeddings
            x = self.layers[0](x) + self.layers[1](positions)  # token_embedding + position_embedding
            x = self.layers[2](x)  # dropout
            
            # Transformer layers
            for layer in self.layers[3:]:
                x = layer(x, attn_mask, key_padding_mask, is_causal)
            
        elif self.rank == self.world_size - 1:
            # Last stage: transformer layers + final norm + output head
            # Transformer layers (all but last 2 modules)
            for layer in self.layers[:-2]:
                x = layer(x, attn_mask, key_padding_mask, is_causal)
            
            # Final norm and output head
            x = self.layers[-2](x)  # final_norm
            x = self.layers[-1](x)  # output_head

        else:
            # Middle stage: transformer layers only
            for layer in self.layers:
                x = layer(x, attn_mask, key_padding_mask, is_causal)
        
        logger.info(f"Rank {self.rank}: forward completed...., output shape: {x.shape}")
        return x

class Pipeline1F1BTrainer:
    """1F1B Pipeline Parallelism with CUDA Streams"""

    def __init__(self, 
                 model: ToyModel,
                 rank: int, 
                 local_rank: int,
                 world_size: int, 
                 num_micro_batches: int,
                 micro_batch_size: int,
                 device: torch.device,
                 loss_fn: Optional[nn.Module] = None,
                 enable_profiling: bool = False):

        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.micro_batch_size = micro_batch_size              # B
        self.seq_len          = model.max_seq_len             # L
        self.d_model          = model.d_model                 # D   (4096 etc.)
        self.activation_shape = torch.Size([self.micro_batch_size, self.seq_len, self.d_model])        
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.device = device
        self.dtype = model.dtype        
        self.enable_profiling = enable_profiling

        # Stream-aware communication handler
        self.comm_handler = StreamAwareAsyncCommHandler(rank, world_size, device)

        # Pipeline state
        self.forward_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.pending_forward_ops: Dict[Any, Tuple[Any, torch.Tensor]] = {}
        self.pending_backward_ops: Dict[Any, Tuple[Any, torch.Tensor]] = {}

        # Ring buffer (one slot per in-flight micro-batch)
        self.pool = min(self.micro_batch_size, self.world_size - 1)
        self.fwd_recv_pool = [
            torch.empty(self.activation_shape, dtype=self.dtype, device=self.device)
            for _ in range(self.pool)
        ]
        self.bwd_recv_pool = [
            torch.empty(self.activation_shape, dtype=self.dtype, device=self.device)
            for _ in range(self.pool)
        ]

        # CUDA events for fine-grained synchronization
        self.forward_comp_events = {}
        self.backward_comp_events = {}

        if self.enable_profiling:
            self.prof = torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                                               on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/rank_{self.rank}'),
                                               record_shapes=True,
                                               with_stack=True
                                            )
            self.prof.start()

        # Statistics
        self.stats = {
            'forward_steps': 0,
            'backward_steps': 0,
            'communication_time': 0.0,
            'computation_time': 0.0,
            'overlap_efficiency': 0.0
        }

        # Model stage
        self.model_stage = self._create_pipeline_stage(model)        

        logger.info(f"Rank {self.rank}: Initialized with streams and pool_size={self.pool_size}")
    
    def _partition_layers(self, total_layers: int) -> Tuple[int, int]:
        """Calculate start and end layer indices for this rank"""
        layers_per_stage = total_layers // self.world_size
        extra_layers = total_layers % self.world_size
        
        if self.rank < extra_layers:
            start = self.rank * (layers_per_stage + 1)
            end = start + layers_per_stage + 1
        else:
            start = extra_layers * (layers_per_stage + 1) + \
                    (self.rank - extra_layers) * layers_per_stage
            end = start + layers_per_stage
        
        return start, end

    def _create_pipeline_stage(self, model: ToyModel) -> PipelineStage:  
        """Create pipeline stage for the rank"""

        # Calculate layers per stage
        start_layer, end_layer = self._partition_layers(total_layers=len(model.layers))

        # Create stage modules
        stage_modules = nn.ModuleList()

        if self.rank == 0:
            # First stage: embeddings + transformer layers
            stage_modules.extend([
                model.token_embedding,
                model.position_embedding,
                model.dropout
            ])
            stage_modules.extend(model.layers[start_layer:end_layer])

        elif self.rank == self.world_size - 1:
            # Last stage: transformer layers + norm layer + output head         
            stage_modules.extend(model.layers[start_layer:end_layer])
            stage_modules.extend([model.final_norm, model.output_head])

        else:
            # Middle stage: transformer layers only
            stage_modules.extend(model.layers[start_layer:end_layer])

        stage = PipelineStage(rank=self.rank, world_size=self.world_size, layers=stage_modules)
        stage = stage.to(self.device)
        
        logger.info(f"Rank {self.rank}: Created stage with {len(stage_modules)} modules, start layer: {start_layer} <-> end layer: {end_layer}")
        
        return stage    

    def forward_step(self, micro_batch_id: int, input_ids: Optional[torch.Tensor] = None,
                     attn_mask: Optional[torch.Tensor] = None,
                     key_padding_mask: Optional[torch.Tensor] = None,
                     is_causal: bool = True) -> Optional[torch.Tensor]:
        """Execute forward pass with stream-based overlap"""
        overlap_start = time.time()
        # --- COMMUNICATION PHASE (on recv_stream) ---
        if self.rank > 0:
            # Check for pre-posted receive
            if micro_batch_id in self.pending_forward_ops:
                recv_req, recv_buffer = self.pending_forward_ops[micro_batch_id]
                # This waits on recv_stream and syncs with default stream
                input_data = self.comm_handler.wait_recv_ready(recv_req, recv_buffer)
                del self.pending_forward_ops[micro_batch_id]
            else:
                # Synchronous receive (fallback)
                buf_idx = micro_batch_id % self.pool_size
                recv_buffer = self.fwd_recv_pool[buf_idx]
                recv_req, _ = self.comm_handler.async_recv_forward(recv_buffer, micro_batch_id)
                if recv_req:
                    input_data = self.comm_handler.wait_recv_ready(recv_req, recv_buffer)
        else:
            if input_ids is None:
                raise ValueError("First stage requires input_ids")
            input_data = input_ids

        comm_time = time.time() - overlap_start
        comp_start = time.time()

        # --- COMPUTATION PHASE (on default stream) ---
        # Record start of computation
        comp_start_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(comp_start_event)

        # Forward pass
        with torch.enable_grad():
            output = self.model_stage(
                input_data,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal
            )
        # Record end of computation
        comp_end_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(comp_end_event)
        self.forward_comp_events[micro_batch_id] = comp_end_event

        comp_time = time.time() - comp_start                