#!/bin/bash
# 快速调试训练（单GPU，小batch）
# 用于测试训练流程是否正常

# 激活虚拟环境
source .venv/bin/activate

echo "🚀 开始快速调试训练..."

# 创建必要的目录
mkdir -p train_log/MP6D/checkpoints
mkdir -p train_log/MP6D/train_info
mkdir -p train_log/MP6D/eval_results

# 单GPU训练，开启debug模式
python3 train_mp6d.py \
    --gpu '0' \
    -debug \
    -lr 1e-2 \
    -epochs 5

echo "✅ 调试训练完成！检查 train_log/MP6D/ 目录下的输出"
