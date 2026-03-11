# Overview

This project implements Pipeline Parallelism with the 1F1B (One Forward One Backward) schedule to train large transformer models efficiently across multiple GPUs. The implementation includes:

- **Custom transformer model** with configurable architecture
- **Asynchronous NCCL communication** for efficient gradient and activation transfer
- **Three-phase 1F1B schedule**: warmup, steady state, and cooldown
- **Multiple versions**: From dummy data to real-world Shakespeare dataset training
- **Docker containerization** for reproducible multi-node training

## What is Pipeline Parallelism?

Pipeline Parallelism is a distributed training technique that splits a neural network model across multiple devices (GPUs) by **layers**. Each device is responsible for a subset of layers, forming a "pipeline stage."

### Benefits:
- **Memory efficiency**: Each GPU only holds a portion of the model
- **Scalability**: Enables training of models too large to fit on a single GPU
- **Throughput**: Multiple micro-batches can be in-flight simultaneously

### Challenges:
- **Pipeline bubbles**: GPUs can be idle during warmup and cooldown phases
- **Communication overhead**: Activations and gradients must be passed between stages
- **Load balancing**: Uneven layer distribution can create bottlenecks

## Setup

### Prerequisites

- NVIDIA GPU with CUDA support (8 GPUs recommended)
- Docker with NVIDIA Container Toolkit
- Python 3.8+
- PyTorch 2.0+

## Performance Considerations

### Factors Affecting Performance

1. **Micro-batch Size**
   - Larger micro-batches → fewer pipeline bubbles
   - Must divide global batch size evenly
   - Limited by memory per GPU

2. **Number of Pipeline Stages**
   - More stages → more parallelism but higher communication overhead
   - Optimal: balance between parallelism and communication

3. **Layer Distribution**
   - Uneven distribution can create bottlenecks
   - First/last stages have additional overhead (embeddings, loss)

4. **Network Bandwidth**
   - NVLINK > PCIe for inter-GPU communication
   - InfiniBand for multi-node training


## References

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Original pipeline parallelism implementation
- [GPipe Paper](https://arxiv.org/abs/1811.06965) - Synchronous pipeline parallelism
- [PipeDream Paper](https://arxiv.org/abs/1806.03377) - Asynchronous pipeline parallelism
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html) - PyTorch distributed training documentation

## License

This project is part of a distributed training research/education initiative.

---

**Happy Training! 🚀**
