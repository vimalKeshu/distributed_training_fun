import logging 
import os 
import time 
import sys
import argparse
import signal


import torch
import torch.nn as nn
import torch.distributed as dist 
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Dict
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncCommHandler:
    """Handles async NCCL communications for pipeline parallelism"""

    def __init__(self, rank:int, world_size:int):
        self.rank = rank 
        self.world_size = world_size
        self.prev_rank = self.rank -  1 if self.rank > 0 else None
        self.next_rank = self.rank +  1 if self.rank < self.world_size - 1 else None

        # Communication tags
        self.FORWARD_TAG_BASE = 1000
        self.BACKWARD_TAG_BASE = 2000
    
    def async_send_forward(self, send_tensor: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], torch.Tensor]:
        """Send tensor to next pipeline stage"""
        logger.debug(f"Rank {self.rank}: About to call dist.isend/irecv/barrier at {time.time()}")
        if self.next_rank is not None:
            tag = self.FORWARD_TAG_BASE + micro_batch_id
            if not send_tensor.is_contiguous():
                send_tensor = send_tensor.contiguous()            
            try:
                req = dist.isend(send_tensor, dst=self.next_rank, tag=tag)
                logger.debug(f"Rank {self.rank}: dist.isend completed successfully")
            except Exception as e:
                logger.error(f"Rank {self.rank}: dist.isend failed: {e}")
                raise            
            logger.debug(f"Rank {self.rank}: Finished dist.isend/irecv/barrier at {time.time()}")
            return req, send_tensor
        logger.debug(f"Rank {self.rank}: Finished dist.isend/irecv/barrier at {time.time()}")
        return None, send_tensor

    def async_recv_forward(self, recv_tensor: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
        """Receive tensor from previous pipeline stage"""
        logger.debug(f"Rank {self.rank}: About to call dist.isend/irecv/barrier at {time.time()}")
        if self.prev_rank is not None:
            try:
                tag = self.FORWARD_TAG_BASE + micro_batch_id
                req = dist.irecv(recv_tensor, src=self.prev_rank, tag=tag)
                logger.debug(f"Rank {self.rank}: dist.irecv completed successfully for tag, {tag} and micro batch id: {micro_batch_id}")
            except Exception as e:
                logger.error(f"Rank {self.rank}: dist.isend failed: {e}")
                raise             
            return req, recv_tensor
        logger.debug(f"Rank {self.rank}: Finished dist.isend/irecv/barrier at {time.time()}")
        return None, None
    
    def async_send_backward(self, send_tensor: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], torch.Tensor]:
        """Send tensor to previous pipeline stage"""
        logger.debug(f"Rank {self.rank}: About to call dist.isend/irecv/barrier at {time.time()}")
        if self.prev_rank is not None:
            tag = self.BACKWARD_TAG_BASE + micro_batch_id
            if not send_tensor.is_contiguous():
                send_tensor = send_tensor.contiguous()                      
            req = dist.isend(send_tensor, dst=self.prev_rank, tag=tag)
            logger.debug(f"Rank {self.rank}: Finished dist.isend/irecv/barrier at {time.time()}")
            return req, send_tensor
        logger.debug(f"Rank {self.rank}: Finished dist.isend/irecv/barrier at {time.time()}")
        return None, send_tensor
    
    def async_recv_backward(self, recv_tensor: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
        """Receive tensor from next pipeline stage"""
        logger.debug(f"Rank {self.rank}: About to call dist.isend/irecv/barrier at {time.time()}")
        if self.next_rank is not None:
            tag = self.BACKWARD_TAG_BASE + micro_batch_id
            req = dist.irecv(recv_tensor, src=self.next_rank, tag=tag)
            logger.debug(f"Rank {self.rank}: Finished dist.isend/irecv/barrier at {time.time()}")
            return req, recv_tensor
        logger.debug(f"Rank {self.rank}: Finished dist.isend/irecv/barrier at {time.time()}")
        return None, None

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
    """1F1B Pipeline Parallelism Trainer"""

    def __init__(self, 
                 model: ToyModel, 
                 rank: int, 
                 local_rank: int,
                 world_size: int, 
                 micro_batch_size: int,
                 device: torch.device,
                 loss_fn: Optional[nn.Module] = None):
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

        # Communication handler
        self.comm_handler = AsyncCommHandler(rank, world_size)

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

        # Statistics
        self.stats = {
            'forward_steps': 0,
            'backward_steps': 0,
            'communication_time': 0.0,
            'computation_time': 0.0
        }

        # Split model into stages
        self.model_stage:PipelineStage = self._create_pipeline_stage(model)

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
    
    def forward_step(self, micro_batch_id: int, input_ids: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, is_causal: bool = True) -> Optional[torch.Tensor]:
        """Execute forward pass for a microbatch"""   
        logger.info(f"Rank {self.rank}: forward_step for a microbatch id: {micro_batch_id} started...") 
        comp_start = time.time()

        # Wait for input from previous stage (if not first stage)
        if self.rank > 0:
            if micro_batch_id in self.pending_forward_ops:
                recv_req, recv_tensor = self.pending_forward_ops[micro_batch_id]
                recv_req.wait()
                input_data = recv_tensor
                del self.pending_forward_ops[micro_batch_id]
            else:
                # Fallback: receive synchronously 
                buf = self.fwd_recv_pool[micro_batch_id % self.pool]
                recv_req, recv_tensor = self.comm_handler.async_recv_forward(buf, micro_batch_id)
                if recv_req:
                    recv_req.wait()
                    input_data = recv_tensor

            input_data.requires_grad_(True)                    
        else:
            input_data = input_ids
        
        # Forward pass - PyTorch automatically tracks operations
        with torch.enable_grad():  # Ensure gradient tracking is enabled
            output = self.model_stage(
                input_data,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal
            )            
        
        logger.info(f"Rank {self.rank}: forward_step for a microbatch id: {micro_batch_id} completed computation...") 
        self.stats['computation_time'] += time.time() - comp_start

        # Cache intermediate activations for backward pass
        self.forward_cache[micro_batch_id] = {
            'input': input_data,
            'output': output
        }        

        comm_start = time.time()
        logger.info(f"Rank {self.rank}: forward_step for a microbatch id: {micro_batch_id} communication started...")

        # Send output to next stage (if not last stage)
        if self.rank < self.world_size - 1:
            send_req, _ = self.comm_handler.async_send_forward(output, micro_batch_id)
            if send_req:
                self.pending_forward_ops[f'send_{micro_batch_id}'] = (send_req, output)

        # Pre-post receive for future forward step
        future_step = micro_batch_id + self.world_size
        if (future_step < self.micro_batch_size and self.rank > 0 and future_step not in self.pending_forward_ops):
            buf = self.fwd_recv_pool[future_step % self.pool]
            recv_req, recv_tensor = self.comm_handler.async_recv_forward(buf, future_step)
            if recv_req:
                self.pending_forward_ops[future_step] = (recv_req, recv_tensor)

        self.stats['communication_time'] += time.time() - comm_start
        self.stats['forward_steps'] += 1
        logger.info(f"Rank {self.rank}: forward_step for a microbatch id: {micro_batch_id} communication ended...") 

    def backward_step(self, micro_batch_id: int, targets: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Execute backward pass for a microbatch"""
        logger.info(f"Rank {self.rank}: backward_step for a microbatch id: {micro_batch_id} computation started...") 
        comp_start = time.time()
        
        # Get cached activations
        if micro_batch_id not in self.forward_cache:
            raise ValueError(f"No forward cache found for microbatch {micro_batch_id}")

        cached_data = self.forward_cache[micro_batch_id]
        input_tensor = cached_data['input']
        output_tensor = cached_data['output']

        # Tensors should already have gradients enabled from forward pass
        assert output_tensor.requires_grad, "Output tensor should already require gradients"
        if self.rank != 0 and input_tensor is not None:
            assert input_tensor.requires_grad, "Input tensor should already require gradients"        

        # Get gradient tensor
        if self.rank == self.world_size - 1:
            # Last stage: compute loss gradient
            if targets is None:
                raise ValueError("Last stage requires targets for loss computation")

            # Reshape for loss computation
            output_flat = output_tensor.view(-1, output_tensor.size(-1))
            targets_flat = targets.view(-1)
            loss = self.loss_fn(output_flat, targets_flat)
            
            # Backward pass starts here
            loss.backward()

        else:
            # Wait for gradient from next stage
            if micro_batch_id in self.pending_backward_ops:
                recv_req, recv_grad = self.pending_backward_ops[micro_batch_id]
                recv_req.wait()
                grad_output = recv_grad
                del self.pending_backward_ops[micro_batch_id]
            else:
                # Fallback: receive synchronously 
                buf = self.bwd_recv_pool[micro_batch_id % self.pool]
                recv_req, recv_grad = self.comm_handler.async_recv_backward(buf, micro_batch_id)
                if recv_req:
                    recv_req.wait()
                    grad_output = recv_grad

            # Backward pass through local stage
            output_tensor.backward(grad_output)
        
        logger.info(f"Rank {self.rank}: backward_step for a microbatch id: {micro_batch_id} computation completed...")
        self.stats['computation_time'] += time.time() - comp_start
        comm_start = time.time()
        logger.info(f"Rank {self.rank}: backward pass for a microbatch id: {micro_batch_id} communication started...")

        # Send gradient to previous stage (if not first stage)
        if self.rank > 0 and input_tensor is not None and input_tensor.grad is not None:
            grad_input = input_tensor.grad    
            send_req, _ = self.comm_handler.async_send_backward(grad_input, micro_batch_id)
            if send_req:
                self.pending_backward_ops[f"send_{micro_batch_id}"] = (send_req, grad_input)

        # Pre-post receive for future backward step
        future_step = micro_batch_id + self.world_size
        if (future_step < self.micro_batch_size and self.rank < self.world_size - 1 and future_step not in self.pending_backward_ops):        
            buf = self.bwd_recv_pool[future_step % self.pool]
            recv_req, recv_grad = self.comm_handler.async_recv_backward(buf, future_step)
            if recv_req:
                self.pending_backward_ops[future_step] = (recv_req, recv_grad)
        
        self.stats['communication_time'] += time.time() - comm_start

        # Clean up cache
        del self.forward_cache[micro_batch_id]
        self.stats['backward_steps'] += 1 
        logger.info(f"Rank {self.rank}: backward pass for a microbatch id: {micro_batch_id} communication ended...") 

    def _wait_pending_operations(self):
        """Wait for all pending operations to complete"""
        for key, (req, tensor) in list(self.pending_forward_ops.items()):
            if isinstance(key, str) and key.startswith('send_'):
                req.wait()
                del self.pending_forward_ops[key]
        
        for key, (req, tensor) in list(self.pending_backward_ops.items()):
            if isinstance(key, str) and key.startswith('send_'):
                req.wait()
                del self.pending_backward_ops[key]

    def schedule_step(self, epoch: int, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Main 1F1B schedule implementation"""

        logger.info(f"Rank {self.rank}: 1F1B schedule started for epoch: {epoch}")

        if (input_ids.shape[0] // self.micro_batch_size) == 0:
            raise ValueError(f"Batch size is zero")
              
        input_micro_batches = list(torch.chunk(input_ids, int(input_ids.shape[0] / self.micro_batch_size), dim=0))
        logger.info(f"Rank {self.rank}: Created {len(input_micro_batches)} micro-batches and shape: {input_micro_batches[0].shape}")
        if targets is not None:
            target_micro_batches = list(torch.chunk(targets, int(input_ids.shape[0] / self.micro_batch_size), dim=0))
        else:
            target_micro_batches = [None] * len(input_micro_batches)     

        num_warmup = min(self.world_size - self.rank - 1, len(input_micro_batches))
        num_steady = max(0, len(input_micro_batches) - num_warmup)

        
        # Phase-1: Warm up (forward passes only)
        for i in range(num_warmup):
            logger.info(f"Rank {self.rank}: Phase-1: Warm up (forward passes only) step: {i}")
            self.forward_step(micro_batch_id=i, 
                              input_ids=input_micro_batches[i])
        
        # Phase 2: 1F1B steady state
        for i in range(num_steady):
            # Forward pass for new microbatch
            forward_step_id = num_warmup + i
            logger.info(f"Rank {self.rank}: Phase 2: 1F1B steady state: Forward pass for new microbatch {forward_step_id}")
            if forward_step_id < len(input_micro_batches):
                self.forward_step(micro_batch_id=forward_step_id, 
                    input_ids=input_micro_batches[forward_step_id])
                
            # Backward pass for oldest cached microbatch
            backward_step_id = i
            logger.info(f"Rank {self.rank}: Phase 2: 1F1B steady state: Backward pass for oldest cached microbatch {forward_step_id}")
            target_data = target_micro_batches[backward_step_id] if target_micro_batches and self.rank == self.world_size - 1 else None
            self.backward_step(micro_batch_id=i, targets=target_data)
        
        # Phase 3: cool down (remaining backwards only) 
        for i in range(num_warmup):
            backward_step_id = num_steady + i
            logger.info(f"Rank {self.rank}: Phase 3: cool down (remaining backwards only) {forward_step_id}")
            target_data = target_micro_batches[backward_step_id] if target_micro_batches and self.rank == self.world_size - 1 else None
            self.backward_step(backward_step_id, target_data)
        
        # Wait for pending communications
        self._wait_pending_operations()
        logger.info(f"Rank {self.rank}: 1F1B schedule ended for epoch: {epoch}")
        return self.stats

def cleanup_handler(signum, frame):
    """Handle cleanup on signal"""
    logger.info("Received signal, cleaning up...")
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass
    sys.exit(0)

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        print("Environment variables RANK and WORLD_SIZE must be set")
        return None, None, None
    
    # Simple sanity check
    assert torch.cuda.device_count() > local_rank, \
        f"Local rank {local_rank} invalid, available GPUs: {torch.cuda.device_count()}"  
     
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=300))
    print(f"Rank {rank}/{world_size} initialized, device: {device}")

    return rank, world_size, local_rank, device 

def create_dummy_data(batch: int, seq: int, vocab: int, device):
    x = torch.randint(0, vocab, (batch, seq), device=device)
    y = torch.randint(0, vocab, (batch, seq), device=device)
    return x, y

def train(args):
    """train the model"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()

    if rank is None:
        raise Exception(f'Not able to get rank value.')    
    
    try:
        # NCCL test
        test_tensor = torch.ones(1, device=device)
        dist.all_reduce(test_tensor)
        logger.info(f"Rank {rank}: Successfully initialized distributed training group...")

        # Create model with validated parameters
        model = ToyModel(
            rank=rank,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dtype=torch.float32
        )

        pipeline:Pipeline1F1BTrainer = Pipeline1F1BTrainer(model=model,
                                                           rank=rank, 
                                                           local_rank=local_rank, 
                                                           world_size=world_size, 
                                                           micro_batch_size=args.micro_batches, 
                                                           device=device)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            pipeline.model_stage.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8
        )        
        # Learning rate scheduler
        total_steps = args.num_steps
        warmup_steps = min(100, total_steps // 10)
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info(f"Rank {rank}: Pipeline setup completed")
        logger.info(f"Rank {rank}: Optimizer: AdamW, LR: {args.learning_rate}, Warmup: {warmup_steps} steps")                

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Rank {rank}: Created model with {total_params:,} parameters")

        start_time = time.time()
        for step in range(args.num_steps):
            step_start_time = time.time()
            # Generate dummy data
            input_ids, targets = create_dummy_data(
                args.batch_size, args.seq_len, args.vocab_size, device
            )           

            logger.info(f"Rank {rank}: dummy input shape: {input_ids.shape}")
            # Zero gradients
            optimizer.zero_grad()
            stats: dict = pipeline.schedule_step(
                epoch=step,
                input_ids=input_ids, 
                targets=targets)

            # Optimizer step
            optimizer.step()
            # Learning rate scheduler step
            scheduler.step()
            step_time = time.time() - step_start_time
            if rank == 0:
                print(f"Iteration {step + 1}: {step_time:.4f}s")
                print(f"  Forward steps: {stats['forward_steps']}")
                print(f"  Backward steps: {stats['backward_steps']}")
                print(f"  Computation time: {stats['computation_time']:.4f}s")
                print(f"  Communication time: {stats['communication_time']:.4f}s")

        # Final synchronization
        dist.barrier()

        total_time = time.time() - start_time
        avg_throughput = (args.num_steps * args.batch_size) / total_time
        
        if rank == 0:
            logger.info("🎉 Training completed successfully!")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Average throughput: {avg_throughput:.1f} samples/s")
            logger.info(f"Steps completed: {args.num_steps}")
    except Exception as e:
        logger.error(f"Rank {rank}: Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info(f"Cleaned up distributed training")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description='1f1b Pipeline Parallel Training')

    # Distributed training arguments
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus-per-node', type=int, default=8, help='GPUs per node')

    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=32, help='Number of transformer layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--seq-len', type=int, default=16, help='Sequence length')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Global batch size')
    parser.add_argument('--micro-batches', type=int, default=4, help='Number of micro-batches per global batch')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num-steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Maximum gradient norm for clipping')

    # NCCL arguments
    parser.add_argument('--timeout-seconds', type=int, default=120, help='NCCL timeout in seconds')

    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N steps')
    parser.add_argument('--save-interval', type=int, default=50, help='Save checkpoint every N steps')

    args = parser.parse_args()       

    # Validate arguments
    if args.batch_size % args.micro_batches != 0:
        raise ValueError("Batch size must be divisible by micro-batches")
    
    if args.d_model % args.n_heads != 0:
        raise ValueError("Model dimension must be divisible by number of heads")

    logger.info("🚀 Starting 1F1B Pipeline Parallel Training")
    logger.info(f"Configuration: {args}")

    train(args)


if __name__ == "__main__":
    main()
