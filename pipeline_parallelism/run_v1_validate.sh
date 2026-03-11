#!/bin/bash

# Training Configuration
NNODES=${PET_NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23456}

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
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  /app/v1_validate.py \
  --nodes $NNODES \
  --gpus-per-node $NPROC_PER_NODE

