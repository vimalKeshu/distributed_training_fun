import logging 
import torch
import torch.nn as nn
import torch.distributed as dist 
from typing import Optional, Tuple, Any, Dict
import time

logger = logging.getLogger(__name__)

class StreamAwareAsyncCommHandler:
    """Handles async NCCL communications with CUDA stream management"""

    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank 
        self.world_size = world_size
        self.device = device
        self.prev_rank = self.rank - 1 if self.rank > 0 else None
        self.next_rank = self.rank + 1 if self.rank < self.world_size - 1 else None

        # Communication tags
        self.FORWARD_TAG_BASE = 1000
        self.BACKWARD_TAG_BASE = 2000
        
        # Create dedicated CUDA streams
        # Default stream (0) is used for computation
        # We create separate streams for communication
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
    
    def async_send_backward(self, send_tensor: torch.Tensor, micro_batch_id: int) -> Tuple[Optional[Any], torch.Tensor]:
        """Send gradient tensor to previous pipeline stage"""
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


class AdvancedPipeline1F1BTrainer:
    """1F1B Pipeline Parallelism with CUDA Streams and Advanced Optimizations"""

    def __init__(self, 
                 model,
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
        self.num_micro_batches = num_micro_batches
        self.micro_batch_size = micro_batch_size
        self.seq_len = model.max_seq_len
        self.d_model = model.d_model
        self.device = device
        self.enable_profiling = enable_profiling
        
        # Activation shape
        self.activation_shape = (self.micro_batch_size, self.seq_len, self.d_model)
        
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        # Stream-aware communication handler
        self.comm_handler = StreamAwareAsyncCommHandler(rank, world_size, device)

        # Pipeline state
        self.forward_cache: Dict[int, Dict[str, Any]] = {}
        self.pending_forward_ops: Dict[int, Tuple[Any, torch.Tensor]] = {}
        self.pending_backward_ops: Dict[int, Tuple[Any, torch.Tensor]] = {}

        # Buffer pools
        self.pool_size = min(self.num_micro_batches, self.world_size - 1)
        
        # Create buffers with pinned memory for faster CPU-GPU transfers
        self.fwd_recv_pool = [
            torch.empty(self.activation_shape, dtype=torch.float32, device=self.device)
            for _ in range(self.pool_size)
        ]
        self.bwd_recv_pool = [
            torch.empty(self.activation_shape, dtype=torch.float32, device=self.device)
            for _ in range(self.pool_size)
        ]

        # CUDA events for fine-grained synchronization
        self.forward_comp_events = {}
        self.backward_comp_events = {}
        
        # Profiling
        if self.enable_profiling:
            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/rank_{rank}'),
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

    def _create_pipeline_stage(self, model):
        """Create pipeline stage (implementation from previous code)"""
        # ... (same as before)
        pass

    def forward_step(self, 
                    micro_batch_id: int, 
                    input_ids: Optional[torch.Tensor] = None,
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

        # Cache for backward
        self.forward_cache[micro_batch_id] = {
            'input': input_data.detach() if input_data.requires_grad else input_data,
            'output': output,
            'comp_event': comp_end_event
        }

        # --- SEND PHASE (on send_stream, overlapped with next computation) ---
        if self.rank < self.world_size - 1:
            # This will wait for computation to finish before sending
            send_req, _ = self.comm_handler.async_send_forward(output, micro_batch_id)

        # --- PRE-POST NEXT RECEIVE (on recv_stream) ---
        # This can start immediately and overlap with current computation
        future_step = micro_batch_id + self.pool_size
        if (future_step < self.num_micro_batches and 
            self.rank > 0 and 
            future_step not in self.pending_forward_ops):
            
            buf_idx = future_step % self.pool_size
            recv_buffer = self.fwd_recv_pool[buf_idx]
            recv_req, _ = self.comm_handler.async_recv_forward(recv_buffer, future_step)
            if recv_req:
                self.pending_forward_ops[future_step] = (recv_req, recv_buffer)

        total_time = time.time() - overlap_start
        overlap_efficiency = 1.0 - (comm_time / total_time) if total_time > 0 else 0
        
        self.stats['communication_time'] += comm_time
        self.stats['computation_time'] += comp_time
        self.stats['overlap_efficiency'] = (self.stats['overlap_efficiency'] * self.stats['forward_steps'] + 
                                           overlap_efficiency) / (self.stats['forward_steps'] + 1)
        self.stats['forward_steps'] += 1
        
        if self.enable_profiling and self.prof:
            self.prof.step()
        
        return output

    def backward_step(self, 
                     micro_batch_id: int, 
                     targets: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Execute backward pass with stream-based overlap"""
        
        overlap_start = time.time()
        
        # Get cached data
        if micro_batch_id not in self.forward_cache:
            raise ValueError(f"No forward cache found for microbatch {micro_batch_id}")

        cached_data = self.forward_cache[micro_batch_id]
        input_tensor = cached_data['input']
        output_tensor = cached_data['output']
        forward_comp_event = cached_data['comp_event']

        # Ensure tensors require grad
        if input_tensor is not None:
            input_tensor.requires_grad_(True)
        output_tensor.requires_grad_(True)

        # --- RECEIVE GRADIENT PHASE ---
        if self.rank == self.world_size - 1:
            # Compute loss
            if targets is None:
                raise ValueError("Last stage requires targets")
            
            # Wait for forward computation to complete
            torch.cuda.current_stream().wait_event(forward_comp_event)
            
            batch_size, seq_len, vocab_size = output_tensor.shape
            loss = self.loss_fn(
                output_tensor.view(-1, vocab_size),
                targets.view(-1)
            )
            grad_output = torch.autograd.grad(loss, output_tensor, retain_graph=False)[0]
            
        else:
            # Receive gradient from next stage
            if micro_batch_id in self.pending_backward_ops:
                recv_req, recv_buffer = self.pending_backward_ops[micro_batch_id]
                grad_output = self.comm_handler.wait_recv_ready(recv_req, recv_buffer)
                del self.pending_backward_ops[micro_batch_id]
            else:
                buf_idx = micro_batch_id % self.pool_size
                recv_buffer = self.bwd_recv_pool[buf_idx]
                recv_req, _ = self.comm_handler.async_recv_backward(recv_buffer, micro_batch_id)
                if recv_req:
                    grad_output = self.comm_handler.wait_recv_ready(recv_req, recv_buffer)

        comm_time = time.time() - overlap_start
        comp_start = time.time()

        # --- BACKWARD COMPUTATION PHASE ---
        # Ensure forward computation is complete
        torch.cuda.current_stream().wait_event(forward_comp_event)
        
        # Record start of backward computation
        backward_start_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(backward_start_event)
        
        # Compute gradients
        if self.rank > 0 and input_tensor is not None:
            grad_input = torch.autograd.grad(
                outputs=output_tensor,
                inputs=input_tensor,
                grad_outputs=grad_output,
                retain_graph=False
            )[0]
        else:
            grad_input = None
        
        # Record end of backward computation
        backward_end_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(backward_end_event)
        self.backward_comp_events[micro_batch_id] = backward_end_event
        
        comp_time = time.time() - comp_start

        # --- SEND GRADIENT PHASE (overlapped) ---
        if self.rank > 0 and grad_input is not None:
            send_req, _ = self.comm_handler.async_send_backward(grad_input, micro_batch_id)

        # --- PRE-POST NEXT BACKWARD RECEIVE ---
        future_step = micro_batch_id + self.pool_size
        if (future_step < self.num_micro_batches and 
            self.rank < self.world_size - 1 and 
            future_step not in self.pending_backward_ops):
            
            buf_idx = future_step % self.pool_size
            recv_buffer = self.bwd_recv_pool[buf_idx]
            recv_req, _ = self.comm_handler.async_recv_backward(recv_buffer, future_step)
            if recv_req:
                self.pending_backward_ops[future_step] = (recv_req, recv_buffer)

        # Update stats
        total_time = time.time() - overlap_start
        self.stats['communication_time'] += comm_time
        self.stats['computation_time'] += comp_time
        self.stats['backward_steps'] += 1

        # Cleanup
        del self.forward_cache[micro_batch_id]
        
        if self.enable_profiling and self.prof:
            self.prof.step()
        
        return grad_input

    def synchronize_streams(self):
        """Ensure all streams are synchronized"""
        # Synchronize all streams
        torch.cuda.current_stream().synchronize()
        self.comm_handler.send_stream.synchronize()
        self.comm_handler.recv_stream.synchronize()
        
    def print_overlap_analysis(self):
        """Print analysis of computation-communication overlap"""
        total_time = self.stats['computation_time'] + self.stats['communication_time']
        theoretical_time = max(self.stats['computation_time'], self.stats['communication_time'])
        actual_overlap = total_time - theoretical_time
        
        logger.info(f"Rank {self.rank} Overlap Analysis:")
        logger.info(f"  Total computation time: {self.stats['computation_time']:.3f}s")
        logger.info(f"  Total communication time: {self.stats['communication_time']:.3f}s")
        logger.info(f"  Theoretical minimum time (perfect overlap): {theoretical_time:.3f}s")
        logger.info(f"  Actual overlap achieved: {actual_overlap:.3f}s")
        logger.info(f"  Average overlap efficiency: {self.stats['overlap_efficiency']:.2%}")
        
    def run_1f1b_schedule_with_streams(self, data_loader, optimizer, grad_clip=1.0):
        """1F1B schedule with stream management"""
        
        # Calculate schedule parameters
        num_warmup = self.world_size - self.rank - 1
        num_steady = self.num_micro_batches - num_warmup
        num_cooldown = self.rank
        
        logger.info(f"Rank {self.rank}: Starting stream-aware 1F1B schedule")
        
        # Reset stats
        self.stats = {k: 0 if not isinstance(v, float) else 0.0 for k, v in self.stats.items()}
        
        # Pre-post initial receives (happens on recv_stream)
        self._pre_post_initial_receives()
        
        # Get micro-batches
        micro_batches = list(data_loader)
        
        iteration_start = time.time()
        
        # === WARMUP PHASE ===
        for i in range(num_warmup):
            if self.rank == 0:
                input_ids, targets = micro_batches[i]
                self.forward_step(i, input_ids=input_ids)
            else:
                self.forward_step(i)
        
        # === STEADY STATE ===
        for i in range(num_warmup, self.num_micro_batches):
            # Forward
            if self.rank == 0:
                input_ids, targets = micro_batches[i]
                self.forward_step(i, input_ids=input_ids)
            else:
                self.forward_step(i)
            
            # Backward (overlapped with next forward)
            backward_id = i - num_warmup
            if self.rank == self.world_size - 1:
                _, targets = micro_batches[backward_id]
                self.backward_step(backward_id, targets=targets)
            else:
                self.backward_step(backward_id)
        
        # === COOLDOWN PHASE ===
        for i in range(num_steady, self.num_micro_batches):
            if self.rank == self.world_size - 1:
                _, targets = micro_batches[i]
                self.backward_step(i, targets=targets)
            else:
                self.backward_step(i)
        
        # Ensure all operations complete
        self.synchronize_streams()
        
        iteration_time = time.time() - iteration_start
        
        # Synchronize before optimizer
        dist.barrier()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model_stage.parameters(), grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Print analysis
        logger.info(f"Rank {self.rank}: Iteration completed in {iteration_time:.3f}s")
        self.print_overlap_analysis()
        
        # Stop profiling if enabled
        if self.enable_profiling and self.prof:
            self.prof.stop()
            
    def _pre_post_initial_receives(self):
        """Pre-post initial receives on recv stream"""
        logger.info(f"Rank {self.rank}: Pre-posting initial receives on recv stream...")
        
        num_warmup_steps = self.world_size - self.rank - 1
        max_prepost = min(num_warmup_steps, self.pool_size)
        
        for step in range(max_prepost):
            # Forward receives
            if self.rank > 0:
                buf_idx = step % self.pool_size
                recv_buffer = self.fwd_recv_pool[buf_idx]
                recv_req, _ = self.comm_handler.async_recv_forward(recv_buffer, step)
                if recv_req:
                    self.pending_forward_ops[step] = (recv_req, recv_buffer)

            # Backward receives
            if self.rank < self.world_size - 1:
                backward_step = step + (self.world_size - self.rank - 1)
                if backward_step < self.num_micro_batches:
                    buf_idx = backward_step % self.pool_size
                    recv_buffer = self.bwd_recv_pool[buf_idx]
                    recv_req, _ = self.comm_handler.async_recv_backward(recv_buffer, backward_step)
                    if recv_req:
                        self.pending_backward_ops[backward_step] = (recv_req, recv_buffer)


# Example usage with profiling
def train_with_streams(rank, world_size, model, data_loader, num_iterations=10):
    """Example training loop with stream-based pipeline parallelism"""
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Create trainer with profiling enabled for first iteration
    trainer = AdvancedPipeline1F1BTrainer(
        model=model,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        num_micro_batches=16,  # Example
        micro_batch_size=8,    # Example
        device=device,
        enable_profiling=(rank == 0)  # Profile only rank 0
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(trainer.model_stage.parameters(), lr=1e-4)
    
    # Training loop
    for iteration in range(num_iterations):
        logger.info(f"Rank {rank}: Starting iteration {iteration}")
        
        # Run one iteration with stream-aware scheduling
        trainer.run_1f1b_schedule_with_streams(
            data_loader=data_loader,
            optimizer=optimizer,
            grad_clip=1.0
        )
        
        # Disable profiling after first iteration
        if iteration == 0 and trainer.enable_profiling:
            trainer.enable_profiling = False
    
    # Cleanup
    dist.destroy_process_group()