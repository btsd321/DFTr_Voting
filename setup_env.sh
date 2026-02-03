#!/bin/bash
# 虚拟环境创建和配置脚本 - 支持迁移到其他机器

set -e  # 遇到错误停止

echo "=========================================="
echo "DFTr_Voting 虚拟环境配置"
echo "=========================================="
echo ""

# ========== 选择环境管理器 ==========
echo "请选择环境管理器:"
echo "  1) Conda/Mamba (推荐，适合深度学习)"
echo "  2) venv + pip (轻量，但需要手动配置 CUDA)"
echo ""
read -p "输入选择 (1/2): " ENV_TYPE

# ========== Conda 环境 ==========
if [ "$ENV_TYPE" = "1" ]; then
    echo ""
    echo "📦 使用 Conda 创建环境..."
    echo ""
    
    # 检查 conda 是否安装
    if ! command -v conda &> /dev/null; then
        echo "❌ Conda 未安装！"
        echo "请先安装 Miniconda 或 Anaconda:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    ENV_NAME="dftr_voting"
    
    # 询问 CUDA 版本
    echo "请选择 CUDA 版本:"
    echo "  1) CUDA 11.3 (RTX 3090/A100 等)"
    echo "  2) CUDA 11.8 (较新的 GPU)"
    echo "  3) CPU only (无 GPU)"
    read -p "输入选择 (1/2/3): " CUDA_VER
    
    # 创建环境
    echo ""
    echo "创建 Conda 环境: $ENV_NAME"
    
    if [ "$CUDA_VER" = "1" ]; then
        # CUDA 11.3
        conda create -n $ENV_NAME python=3.8 -y
        conda activate $ENV_NAME
        conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y
    elif [ "$CUDA_VER" = "2" ]; then
        # CUDA 11.8
        conda create -n $ENV_NAME python=3.8 -y
        conda activate $ENV_NAME
        conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        # CPU only
        conda create -n $ENV_NAME python=3.8 -y
        conda activate $ENV_NAME
        conda install pytorch torchvision cpuonly -c pytorch -y
    fi
    
    # 安装其他依赖
    echo ""
    echo "安装项目依赖..."
    conda install numpy scipy scikit-learn opencv pillow pyyaml h5py matplotlib seaborn cython tqdm -y
    
    pip install -r requirements.txt
    
    # 编译 RandLA-Net CUDA 算子
    echo ""
    echo "编译 RandLA-Net CUDA 算子..."
    cd models/RandLA
    bash compile_op.sh
    cd ../..
    
    echo ""
    echo "=========================================="
    echo "✅ Conda 环境创建成功！"
    echo ""
    echo "激活环境:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "导出环境（迁移到其他机器）:"
    echo "  conda env export > environment_exact.yml"
    echo ""
    echo "在其他机器上恢复:"
    echo "  conda env create -f environment_exact.yml"
    echo "=========================================="

# ========== venv 环境 ==========
elif [ "$ENV_TYPE" = "2" ]; then
    echo ""
    echo "📦 使用 venv 创建环境..."
    echo ""
    
    ENV_DIR="venv_dftr"
    
    # 创建虚拟环境
    python3 -m venv $ENV_DIR
    source $ENV_DIR/bin/activate
    
    # 升级 pip
    pip install --upgrade pip setuptools wheel
    
    # 询问 PyTorch 版本
    echo "请选择 PyTorch 版本:"
    echo "  1) CUDA 11.3"
    echo "  2) CUDA 11.8"
    echo "  3) CPU only"
    read -p "输入选择 (1/2/3): " TORCH_VER
    
    if [ "$TORCH_VER" = "1" ]; then
        pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    elif [ "$TORCH_VER" = "2" ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision
    fi
    
    # 安装其他依赖
    pip install -r requirements.txt
    
    # 编译 RandLA-Net
    echo ""
    echo "编译 RandLA-Net CUDA 算子..."
    cd models/RandLA
    bash compile_op.sh
    cd ../..
    
    echo ""
    echo "=========================================="
    echo "✅ venv 环境创建成功！"
    echo ""
    echo "激活环境:"
    echo "  source $ENV_DIR/bin/activate"
    echo ""
    echo "导出依赖（迁移到其他机器）:"
    echo "  pip freeze > requirements_exact.txt"
    echo ""
    echo "在其他机器上恢复:"
    echo "  python3 -m venv venv_dftr"
    echo "  source venv_dftr/bin/activate"
    echo "  pip install -r requirements_exact.txt"
    echo "=========================================="
    
else
    echo "❌ 无效选择！"
    exit 1
fi

echo ""
echo "🎉 环境配置完成！"
echo ""
echo "下一步:"
echo "  1. 激活环境"
echo "  2. 配置数据集: ./setup_mp6d_data.sh"
echo "  3. 验证环境: python3 verify_setup.py"
echo "  4. 开始训练: ./train_mp6d.sh"
