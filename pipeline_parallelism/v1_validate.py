import logging 
import os 
import time 
import argparse
import signal
import random

import torch
import torch.nn as nn 

from v1_1f1b import *


def set_determinism():
    SEED = 42
    import numpy as np
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np = __import__("numpy")
    np.random.seed(SEED)

class ExtendCommunicationHandler(AsyncCommHandler):

    def async_send_params(self, send_tensor: torch.Tensor, target_rank: int, tag: int) -> Tuple[Optional[Any], torch.Tensor]:
        if target_rank is not None:
            if not send_tensor.is_contiguous():
                send_tensor = send_tensor.contiguous()            
            try:
                req = dist.isend(send_tensor, dst=target_rank, tag=tag)
            except Exception as e:
                logger.error(f"Rank {self.rank}: dist.isend failed: {e}")
                raise            
            return req, send_tensor
        return None, send_tensor

    def async_recv_params(self, recv_tensor: torch.Tensor, source_rank: int, tag: int) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
        if source_rank is not None:
            try:
                req = dist.irecv(recv_tensor, src=source_rank, tag=tag)
            except Exception as e:
                logger.error(f"Rank {self.rank}: dist.isend failed: {e}")
                raise             
            return req, recv_tensor
        return None, None
    
class DPToyModel(ToyModel):
    def forward(self, 
                x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        # Token embeddings + position embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)  # token_embedding + position_embedding
        x = self.dropout(x)  # dropout        

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask, is_causal)

        # Final norm and output head
        x = self.final_norm(x)  # final_norm
        x = self.output_head(x)  # output_head

        return x

def get_module_params_grad(modules: nn.Module | nn.ModuleList) -> Dict[str,  Optional[torch.Tensor]]:
    params:Dict[str, Optional[torch.Tensor]] = {}
    if isinstance(modules, nn.Module):
        for n, p in modules.named_parameters():
            if p.grad is None:
                params[n] = None
            else:
                params[n] = p.grad.detach().float().cpu().clone()
    else:
        for module in modules:
            for n, p in module.named_parameters():
                if p.grad is None:
                    params[n] = None
                else:
                    params[n] = p.grad.detach().float().cpu().clone()            
    return params

def get_module_params(modules: nn.Module | nn.ModuleList) -> Dict[str, torch.Tensor]:
    params:Dict[str, torch.Tensor] = {}
    if isinstance(modules, nn.Module):
        for k, v in modules.state_dict().items():
            params[k] = v.detach().float().cpu().clone()
    else:
        for module in modules:
            for k, v in module.state_dict().items():
                params[k] = v.detach().float().cpu().clone()            
    return params


def train_reference(args, rank, world_size, local_rank, device) -> Optional[List[Dict[str, Any]]]:
    """train reference model"""
    if rank != 0 and local_rank !=0:
        return None
    
    set_determinism()

    model = DPToyModel(rank=0,
                        vocab_size=args.vocab_size,
                        d_model=args.d_model,
                        n_layers=args.n_layers,
                        n_heads=args.n_heads,
                        d_ff=args.d_ff,
                        max_seq_len=args.seq_len).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created model with {total_params:,} parameters")    
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    per_step:List[Dict[str,Any]] = []
    for step in range(args.num_steps):
        # Generate dummy data
        input_ids, targets = create_dummy_data(args.batch_size, args.seq_len, args.vocab_size, device) 
        optimizer.zero_grad()

        output = model(input_ids)
        loss = loss_fn(output.view(-1, output.size(-1)), targets.view(-1))
        logger.info(f"Reference model: loss: {loss:0.4f}")
        loss.backward()

        # capture grads before update
        grads_before = get_module_params_grad(model)
        params_before = get_module_params(model)

        optimizer.step()
        scheduler.step()

        params_after = get_module_params(model)

        per_step.append({
            "step": step,
            "loss": loss.detach().float().cpu().clone(),
            "grads": grads_before,
            "params_before": params_before,
            "params_after": params_after
        })

        # clean up
        torch.cuda.empty_cache()
        del model

        return per_step        

def train_1f1b(args, rank, world_size, local_rank, device) -> List[Dict[str, Any]]:
    """validate 1f1b pp training model"""    
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
        per_step = []
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

            # capture grads before update
            grads_before = get_module_params_grad(pipeline.model_stage.layers)
            params_before = get_module_params(pipeline.model_stage.layers)

            optimizer.step()
            scheduler.step()

            params_after = get_module_params(model)

            per_step.append({
                "step": step,
                "loss": None,
                "grads": grads_before,
                "params_before": params_before,
                "params_after": params_after
            })

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

def validate():
    parser = argparse.ArgumentParser(description='1f1b Pipeline Parallel Training')

    # Distributed training arguments
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus-per-node', type=int, default=8, help='GPUs per node')

    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=32, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=16, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n-heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=32, help='Feed-forward dimension')
    parser.add_argument('--seq-len', type=int, default=8, help='Sequence length')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4, help='Global batch size')
    parser.add_argument('--micro-batches', type=int, default=2, help='Number of micro-batches per global batch')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num-steps', type=int, default=3, help='Number of training steps')
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

    # Setup signal handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    if rank is None:
        raise Exception(f'Not able to get rank value.') 
    if rank == 0:
        logger.info("🚀 Starting 1F1B Single GPU Training")
        logger.info(f"Configuration: {args}")
        reference_model_metrics:Optional[List[Dict[str,Any]]] = train_reference(args, rank, world_size, local_rank, device)
        if reference_model_metrics:
            torch.save(reference_model_metrics, "reference_model_metrics.pt")
            logger.info(f"Saved reference model metrics !!!")

    dist.barrier()
    pp_model_metrics:Optional[List[Dict[str,Any]]] = train_1f1b(args, rank, world_size, local_rank, device)
    dist.barrier()
    if rank != 0:
        reference_model_metrics:Optional[List[Dict[str,Any]]] = torch.load("reference_model_metrics.pt", map_location="cpu")
        logger.info(f"Loaded reference model metrics !!!")
    
    # Compare
    logger.info("Done........")


if __name__ == "__main__":
    validate()
