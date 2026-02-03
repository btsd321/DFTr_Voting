# 🚀 DFTr_Voting 虚拟环境和项目迁移指南

本指南介绍如何创建可迁移的虚拟环境，以便在不同机器上运行本项目。

---

## 📋 目录
1. [快速开始](#快速开始)
2. [Conda 环境（推荐）](#conda-环境推荐)
3. [venv 环境（轻量）](#venv-环境轻量)
4. [Docker 容器（最佳可移植性）](#docker-容器最佳可移植性)
5. [迁移到其他机器](#迁移到其他机器)
6. [常见问题](#常见问题)

---

## 🎯 快速开始

```bash
# 1. 运行自动配置脚本
./setup_env.sh

# 2. 按提示选择：
#    - 环境管理器（Conda 或 venv）
#    - CUDA 版本（11.3, 11.8 或 CPU）

# 3. 激活环境后，配置数据集
./setup_mp6d_data.sh

# 4. 验证环境
python3 verify_setup.py
```

---

## 🐍 Conda 环境（推荐）

### 优点
- ✅ 自动处理 CUDA/cuDNN 依赖
- ✅ 环境隔离更彻底
- ✅ 易于导出和恢复
- ✅ 适合深度学习项目

### 创建环境

#### 方法 1: 使用 environment.yml（推荐）

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate dftr_voting

# 编译 CUDA 算子
cd models/RandLA
bash compile_op.sh
cd ../..
```

#### 方法 2: 手动创建

```bash
# 创建 Python 3.8 环境
conda create -n dftr_voting python=3.8 -y

# 激活环境
conda activate dftr_voting

# 安装 PyTorch (CUDA 11.3)
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y

# 或 CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他依赖
conda install numpy scipy opencv pillow pyyaml matplotlib -y
pip install -r requirements.txt
```

### 导出环境（用于迁移）

```bash
# 方法 1: 导出完整环境（精确版本）
conda env export > environment_exact.yml

# 方法 2: 导出跨平台环境（不含 build 信息）
conda env export --no-builds > environment_cross_platform.yml

# 方法 3: 只导出手动安装的包
conda env export --from-history > environment_minimal.yml
```

### 打包整个环境（离线迁移）

```bash
# 1. 打包环境
conda pack -n dftr_voting -o dftr_voting_env.tar.gz

# 2. 传输到目标机器
scp dftr_voting_env.tar.gz user@target-machine:/path/to/

# 3. 在目标机器上解包
mkdir -p ~/miniconda3/envs/dftr_voting
tar -xzf dftr_voting_env.tar.gz -C ~/miniconda3/envs/dftr_voting

# 4. 激活环境
conda activate dftr_voting
```

---

## 📦 venv 环境（轻量）

### 优点
- ✅ Python 内置，无需额外安装
- ✅ 占用空间小
- ✅ 适合熟悉 pip 的用户

### 缺点
- ⚠️ 需要手动安装 CUDA/cuDNN
- ⚠️ 环境迁移稍复杂

### 创建环境

```bash
# 1. 创建虚拟环境
python3 -m venv venv_dftr

# 2. 激活环境
source venv_dftr/bin/activate  # Linux/Mac
# 或
venv_dftr\Scripts\activate     # Windows

# 3. 升级 pip
pip install --upgrade pip setuptools wheel

# 4. 安装 PyTorch (根据 CUDA 版本)
# CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 11.8
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision

# 5. 安装项目依赖
pip install -r requirements.txt

# 6. 编译 CUDA 算子
cd models/RandLA
bash compile_op.sh
cd ../..
```

### 导出环境

```bash
# 导出所有已安装的包
pip freeze > requirements_exact.txt

# 或使用 pipreqs 只导出项目实际使用的包（推荐）
pip install pipreqs
pipreqs . --force
```

### 打包环境（离线迁移）

```bash
# 1. 下载所有依赖的 wheel 文件
pip download -r requirements.txt -d packages/

# 2. 打包整个目录
tar -czf dftr_venv_packages.tar.gz packages/ requirements.txt

# 3. 在目标机器上安装
tar -xzf dftr_venv_packages.tar.gz
python3 -m venv venv_dftr
source venv_dftr/bin/activate
pip install --no-index --find-links=packages -r requirements.txt
```

---

## 🐳 Docker 容器（最佳可移植性）

### 创建 Dockerfile

```dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip git wget \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace/DFTr_Voting

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN pip3 install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install --no-cache-dir -r requirements.txt

# 编译 CUDA 算子
RUN cd models/RandLA && bash compile_op.sh && cd ../..

# 设置入口点
CMD ["/bin/bash"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t dftr_voting:latest .

# 运行容器
docker run --gpus all -it --rm \
    -v /path/to/datasets:/workspace/DFTr_Voting/datasets \
    -v /path/to/train_log:/workspace/DFTr_Voting/train_log \
    dftr_voting:latest

# 保存镜像（迁移到其他机器）
docker save dftr_voting:latest | gzip > dftr_voting_docker.tar.gz

# 在目标机器加载
gunzip -c dftr_voting_docker.tar.gz | docker load
```

---

## 🚚 迁移到其他机器

### 场景 1: 有网络连接

**Conda 环境**:
```bash
# 源机器
conda env export > environment.yml
scp environment.yml user@target:/path/

# 目标机器
conda env create -f environment.yml
conda activate dftr_voting
```

**venv 环境**:
```bash
# 源机器
pip freeze > requirements.txt
scp requirements.txt user@target:/path/

# 目标机器
python3 -m venv venv_dftr
source venv_dftr/bin/activate
pip install -r requirements.txt
```

### 场景 2: 无网络连接（离线）

**Conda 离线包**:
```bash
# 源机器
conda install conda-pack
conda pack -n dftr_voting -o dftr_env.tar.gz

# 传输到目标机器
scp dftr_env.tar.gz user@target:/path/

# 目标机器
mkdir -p ~/miniconda3/envs/dftr_voting
tar -xzf dftr_env.tar.gz -C ~/miniconda3/envs/dftr_voting
conda activate dftr_voting
conda-unpack  # 激活脚本
```

**pip 离线包**:
```bash
# 源机器
pip download -r requirements.txt -d pip_packages/
tar -czf pip_packages.tar.gz pip_packages/ requirements.txt

# 目标机器
tar -xzf pip_packages.tar.gz
python3 -m venv venv_dftr
source venv_dftr/bin/activate
pip install --no-index --find-links=pip_packages -r requirements.txt
```

### 场景 3: 完整项目迁移

```bash
# 1. 打包整个项目（排除大文件）
tar -czf dftr_voting_project.tar.gz \
    --exclude='datasets' \
    --exclude='train_log' \
    --exclude='*.pth.tar' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    DFTr_Voting/

# 2. 单独打包预训练模型
tar -czf pretrained_models.tar.gz train_log/MP6D/checkpoints/

# 3. 传输文件
scp dftr_voting_project.tar.gz user@target:/path/
scp pretrained_models.tar.gz user@target:/path/

# 4. 在目标机器解压
tar -xzf dftr_voting_project.tar.gz
tar -xzf pretrained_models.tar.gz -C DFTr_Voting/

# 5. 配置环境和数据集
cd DFTr_Voting
./setup_env.sh
./setup_mp6d_data.sh
```

---

## ❓ 常见问题

### Q1: CUDA 版本不匹配
```bash
# 检查系统 CUDA 版本
nvidia-smi  # 查看 CUDA Version

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配的 PyTorch
conda install pytorch cudatoolkit=<version> -c pytorch
```

### Q2: RandLA-Net 编译失败
```bash
# 确保安装了编译工具
sudo apt install build-essential

# 检查 CUDA 路径
echo $CUDA_HOME  # 应该指向 CUDA 安装目录

# 手动设置
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新编译
cd models/RandLA
bash compile_op.sh
```

### Q3: 导入模块失败
```bash
# 确保激活了虚拟环境
which python  # 应该指向虚拟环境的 python

# 检查包是否安装
pip list | grep torch
pip list | grep opencv

# 重新安装
pip install -r requirements.txt --force-reinstall
```

### Q4: 不同操作系统迁移
- Linux → Linux: ✅ 直接迁移
- Linux → Windows: ⚠️ 需要重新编译 CUDA 算子
- Windows → Linux: ⚠️ 需要重新编译 CUDA 算子
- 建议: 使用 Docker 实现跨平台

---

## 📝 检查清单

迁移前确认：
- [ ] 环境文件已导出 (`environment.yml` 或 `requirements.txt`)
- [ ] 预训练模型已备份
- [ ] CUDA 版本已确认
- [ ] 数据集路径已更新
- [ ] CUDA 算子已重新编译（如果跨机器）

迁移后验证：
```bash
# 1. 检查 Python 环境
python --version

# 2. 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 3. 检查项目依赖
python verify_setup.py

# 4. 运行测试
python train_mp6d.py -debug
```

---

## 🔗 相关资源

- **Conda Pack**: https://conda.github.io/conda-pack/
- **PyTorch 安装**: https://pytorch.org/get-started/locally/
- **Docker Hub**: https://hub.docker.com/r/nvidia/cuda

---

**完成配置后，运行 `python verify_setup.py` 验证环境！**
