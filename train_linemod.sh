#!/bin/bash
# Training script for Linemod dataset
# Usage: ./train_linemod.sh [object_name]
# Example: ./train_linemod.sh ape

# 激活虚拟环境
source .venv/bin/activate

# Default object (if not specified)
OBJ=${1:-ape}

# Available objects: ape, benchvise, cam, can, cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone

echo "Training on Linemod object: $OBJ"

# Set GPU count
n_gpu=2

# Distributed training with 2 GPUs
python3 -m torch.distributed.launch \
    --nproc_per_node=$n_gpu \
    train_linemod.py \
    --gpus=$n_gpu \
    --cls=$OBJ \
    --gpu='0,1'

# For single GPU training, use:
# python3 train_linemod.py --cls=$OBJ --gpu='0'
