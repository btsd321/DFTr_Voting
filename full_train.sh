#!/bin/bash
# 完整训练脚本（多GPU分布式训练）
# 在快速调试成功后使用

# 激活虚拟环境
source .venv/bin/activate

n_gpu=2  # 使用的GPU数量，根据你的硬件调整
master_port=5235  # 分布式训练端口

echo "🚀 开始完整训练（${n_gpu} GPUs）..."

# 创建必要的目录
mkdir -p train_log/MP6D/checkpoints
mkdir -p train_log/MP6D/train_info
mkdir -p train_log/MP6D/eval_results

# 分布式训练
python3 -m torch.distributed.launch \
    --nproc_per_node=$n_gpu \
    --master_port $master_port \
    train_mp6d.py \
    --gpus=$n_gpu \
    --gpu='0,1' \
    -lr 1e-2 \
    -epochs 1000

# 如果要从checkpoint恢复训练，使用以下命令：
# --checkpoint="train_log/MP6D/checkpoints/FFB6D_epoch_XX.pth.tar"

echo "✅ 训练完成！"
echo "📊 查看训练日志: tensorboard --logdir train_log/MP6D/train_info"
echo "💾 模型保存在: train_log/MP6D/checkpoints/"
