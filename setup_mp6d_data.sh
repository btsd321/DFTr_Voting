#!/bin/bash
# MP6D 数据集解压和放置脚本

set -e  # 遇到错误立即停止
# 激活虚拟环境（如果需要Python工具）
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
echo "=========================================="
echo "MP6D 数据集解压和配置"
echo "=========================================="
echo ""

PROJECT_ROOT="/home/lixinlong/Project/DFTr_Voting"
DOWNLOAD_DIR="$PROJECT_ROOT/MP6D"
TARGET_DIR="$PROJECT_ROOT/datasets/MP6D"

# 1. 检查下载的分卷压缩包
echo "📦 步骤 1: 检查下载的压缩包..."
cd $DOWNLOAD_DIR
if [ ! -f "data.tar.gz00" ]; then
    echo "❌ 错误: 未找到 data.tar.gz00"
    echo "请确保已下载完整的 MP6D 数据集"
    exit 1
fi

echo "找到以下分卷压缩包:"
ls -lh data*.tar.gz* 2>/dev/null | wc -l
echo ""

# 2. 合并并解压 data.tar.gz (真实数据)
echo "📂 步骤 2: 解压真实数据 (data.tar.gz)..."
if [ -d "$DOWNLOAD_DIR/data_real" ]; then
    echo "⚠️  data_real 目录已存在，跳过解压"
else
    echo "合并分卷压缩包 data.tar.gz..."
    cat data.tar.gz* > data.tar.gz
    
    echo "解压 data.tar.gz (可能需要几分钟)..."
    tar -xzf data.tar.gz
    
    # 重命名解压后的目录为 data_real
    if [ -d "data" ]; then
        mv data data_real
    fi
    
    echo "✅ 真实数据解压完成"
fi
echo ""

# 3. 解压合成数据 (可选，训练时需要)
echo "📂 步骤 3: 解压合成数据 (data_syn_1 & data_syn_2)..."

# data_syn_1
if [ -d "$DOWNLOAD_DIR/data_syn_1" ]; then
    echo "⚠️  data_syn_1 目录已存在，跳过解压"
else
    if [ -f "data_syn_1.tar.gz00" ]; then
        echo "合并 data_syn_1 分卷..."
        cat data_syn_1.tar.gz* > data_syn_1.tar.gz
        
        echo "解压 data_syn_1.tar.gz..."
        tar -xzf data_syn_1.tar.gz
        echo "✅ data_syn_1 解压完成"
    else
        echo "⚠️  未找到 data_syn_1，跳过"
    fi
fi

# data_syn_2
if [ -d "$DOWNLOAD_DIR/data_syn_2" ]; then
    echo "⚠️  data_syn_2 目录已存在，跳过解压"
else
    if [ -f "data_syn_2.tar.gz00" ]; then
        echo "合并 data_syn_2 分卷..."
        cat data_syn_2.tar.gz* > data_syn_2.tar.gz
        
        echo "解压 data_syn_2.tar.gz..."
        tar -xzf data_syn_2.tar.gz
        echo "✅ data_syn_2 解压完成"
    else
        echo "⚠️  未找到 data_syn_2，跳过"
    fi
fi
echo ""

# 4. 移动到正确的位置
echo "📁 步骤 4: 移动数据到项目目录..."
mkdir -p $TARGET_DIR

# 移动真实数据
if [ -d "$DOWNLOAD_DIR/data_real" ]; then
    echo "移动 data_real -> $TARGET_DIR/"
    mv $DOWNLOAD_DIR/data_real $TARGET_DIR/
else
    echo "⚠️  data_real 目录不存在"
fi

# 移动合成数据
if [ -d "$DOWNLOAD_DIR/data_syn_1" ]; then
    echo "移动 data_syn_1 -> $TARGET_DIR/"
    mv $DOWNLOAD_DIR/data_syn_1 $TARGET_DIR/
fi

if [ -d "$DOWNLOAD_DIR/data_syn_2" ]; then
    echo "移动 data_syn_2 -> $TARGET_DIR/"
    mv $DOWNLOAD_DIR/data_syn_2 $TARGET_DIR/
fi
echo ""

# 5. 清理临时文件（可选）
echo "🧹 步骤 5: 清理临时文件..."
read -p "是否删除已解压的 tar.gz 文件以节省空间? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd $DOWNLOAD_DIR
    rm -f data.tar.gz data_syn_1.tar.gz data_syn_2.tar.gz
    echo "✅ 临时文件已删除"
else
    echo "保留临时文件"
fi
echo ""

# 6. 验证目录结构
echo "✅ 步骤 6: 验证数据集结构..."
echo "目标目录: $TARGET_DIR"
ls -lh $TARGET_DIR/
echo ""

echo "=========================================="
echo "🎉 MP6D 数据集配置完成！"
echo ""
echo "数据集位置:"
echo "  - 真实数据: $TARGET_DIR/data_real/"
echo "  - 合成数据1: $TARGET_DIR/data_syn_1/"
echo "  - 合成数据2: $TARGET_DIR/data_syn_2/"
echo ""
echo "下一步:"
echo "  1. 验证环境: python3 verify_setup.py"
echo "  2. 训练模型: ./train_mp6d.sh"
echo "=========================================="
