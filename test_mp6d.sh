#!/bin/bash
# 测试/推理脚本
# 使用方法: 
# 1. 修改 tst_mdl 为你的checkpoint路径
# 2. 调整 --gpu 参数选择使用的GPU

# 激活虚拟环境
source .venv/bin/activate

# checkpoint路径（修改为你实际的模型路径）
tst_mdl="train_log/MP6D/checkpoints/FFB6D_best.pth.tar"

# 检查checkpoint是否存在
if [ ! -f "$tst_mdl" ]; then
    echo "❌ 错误: Checkpoint 不存在: $tst_mdl"
    echo "📝 请修改 tst_mdl 变量为正确的checkpoint路径"
    echo "💡 可用的checkpoints:"
    ls -lh train_log/MP6D/checkpoints/*.pth.tar 2>/dev/null || echo "   (暂无checkpoint)"
    exit 1
fi

echo "🧪 开始测试..."
echo "📦 使用模型: $tst_mdl"

python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 14152 \
    train_mp6d.py \
    --gpu '0' \
    -eval_net \
    -checkpoint $tst_mdl \
    -test \
    -test_pose
    # 添加 -debug 参数可以启用调试模式

echo "✅ 测试完成！"
echo "📊 评估结果保存在: train_log/MP6D/eval_results/"
