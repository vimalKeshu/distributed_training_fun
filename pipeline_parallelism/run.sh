#!/bin/bash

# Training Configuration
NNODES=${PET_NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23456}

# Model configuration
BATCH_SIZE=${BATCH_SIZE:-64}
MICRO_BATCHES=${MICRO_BATCHES:-16}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
NUM_STEPS=${NUM_STEPS:-100}

echo "=== Torchrun Multi-Node Pipeline Training ==="
echo "Nodes: $NNODES"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Batch size: $BATCH_SIZE"
echo "Micro batches: $MICRO_BATCHES"
echo "LEARNING RATE: $LEARNING_RATE"
echo "NUM STEPS: $NUM_STEPS"
echo "============================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=ERROR
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_DISTRIBUTED_DEBUG=DETAIL 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Launch training
torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$RANK \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  /app/1f1b.py \
  --nodes $NNODES \
  --gpus-per-node $NPROC_PER_NODE \
  --batch-size $BATCH_SIZE \
  --micro-batches $MICRO_BATCHES \
  --learning-rate $LEARNING_RATE \
  --num-steps $NUM_STEPS

