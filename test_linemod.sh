#!/bin/bash
# Testing script for Linemod dataset
# Usage: ./test_linemod.sh [object_name] [checkpoint_path]
# Example: ./test_linemod.sh ape train_log/linemod/checkpoints/ape_best.pth.tar

OBJ=${1:-ape}
CHECKPOINT=${2:-"train_log/linemod/checkpoints/${OBJ}_best.pth.tar"}

echo "Testing on Linemod object: $OBJ"
echo "Using checkpoint: $CHECKPOINT"

# Test with pose estimation
python3 train_linemod.py \
    --cls=$OBJ \
    -eval_net \
    -checkpoint $CHECKPOINT \
    -test \
    -test_pose

# For calculating metrics only
# python3 train_linemod.py --cls=$OBJ -cal_metrics
