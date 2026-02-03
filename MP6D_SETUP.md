# MP6D 数据集配置指南

## 📦 你下载的文件

在 `/home/lixinlong/Project/DFTr_Voting/MP6D/` 目录中：

```
├── data.tar.gz00 ~ data.tar.gz13        # 真实数据 (约 14GB)
├── data_syn_1.tar.gz00 ~ .gz07          # 合成数据1 (约 16GB)
├── data_syn_2.tar.gz00 ~ .gz07          # 合成数据2 (约 15GB)
├── models_cad.tar.gz00                  # 3D CAD 模型
└── FFB6D_best.pth.tar                   # 预训练模型 (389MB)
```

## 🚀 快速开始（一键配置）

```bash
# 运行自动配置脚本
./setup_mp6d_data.sh
```

脚本会自动：
1. ✅ 合并分卷压缩包
2. ✅ 解压所有数据集
3. ✅ 移动到正确的 `datasets/MP6D/` 目录
4. ✅ 清理临时文件（可选）

---

## 📋 手动配置（如果自动脚本失败）

### 步骤 1: 合并并解压真实数据

```bash
cd /home/lixinlong/Project/DFTr_Voting/MP6D

# 合并分卷（data.tar.gz 有 14 个分卷）
cat data.tar.gz* > data.tar.gz

# 解压
tar -xzf data.tar.gz

# 移动到项目目录
mv data datasets/MP6D/data_real
```

### 步骤 2: 解压合成数据（用于训练）

```bash
# 合成数据 1
cat data_syn_1.tar.gz* > data_syn_1.tar.gz
tar -xzf data_syn_1.tar.gz
mv data_syn_1 ../datasets/MP6D/

# 合成数据 2
cat data_syn_2.tar.gz* > data_syn_2.tar.gz
tar -xzf data_syn_2.tar.gz
mv data_syn_2 ../datasets/MP6D/
```

### 步骤 3: 解压 CAD 模型

```bash
# 如果有多个分卷，先合并
cat models_cad.tar.gz* > models_cad.tar.gz
tar -xzf models_cad.tar.gz
mv models_cad ../datasets/MP6D/
```

### 步骤 4: 放置预训练模型

```bash
# 创建模型目录
mkdir -p ../train_log/MP6D/checkpoints/

# 移动预训练权重
cp FFB6D_best.pth.tar ../train_log/MP6D/checkpoints/
```

---

## 📁 最终目录结构

配置完成后，项目结构应该是：

```
DFTr_Voting/
├── datasets/
│   └── MP6D/
│       ├── data_real/          # 真实测试数据
│       │   ├── color/
│       │   ├── depth/
│       │   ├── label/
│       │   └── ...
│       ├── data_syn_1/         # 合成训练数据 1
│       ├── data_syn_2/         # 合成训练数据 2
│       ├── models_cad/         # 3D 物体模型
│       ├── MP6D_dataset_ori.py
│       ├── MP6D_kps/           # 关键点定义
│       └── dataset_config/
└── train_log/
    └── MP6D/
        └── checkpoints/
            └── FFB6D_best.pth.tar  # 预训练模型
```

---

## ⚙️ 配置验证

运行验证脚本检查配置：

```bash
python3 verify_setup.py
```

预期输出：
```
✅ MP6D 数据集根目录
✅ 找到真实数据: data_real
✅ 找到合成数据: data_syn_1, data_syn_2
✅ CAD 模型目录
✅ 预训练模型
```

---

## 💾 磁盘空间需求

- **压缩包总计**: ~45 GB
- **解压后**: ~60 GB
- **建议**: 解压后可删除 `.tar.gz` 文件节省空间

删除压缩包：
```bash
cd /home/lixinlong/Project/DFTr_Voting/MP6D
rm -f *.tar.gz  # 谨慎操作！确保已成功解压
```

---

## 🏋️ 开始训练

配置完成后：

```bash
# 方式 1: 使用预训练模型微调
./train_mp6d.sh -checkpoint train_log/MP6D/checkpoints/FFB6D_best.pth.tar

# 方式 2: 从头训练
./train_mp6d.sh

# Debug 模式（快速验证流程）
python3 train_mp6d.py -debug
```

---

## 🧪 测试模型

```bash
./test_mp6d.sh
```

或指定模型路径：
```bash
python3 train_mp6d.py \
    -eval_net \
    -checkpoint train_log/MP6D/checkpoints/FFB6D_best.pth.tar \
    -test -test_pose
```

---

## ❓ 常见问题

### Q1: 分卷压缩包合并失败
```bash
# 确保所有分卷都下载完整
ls -lh MP6D/data.tar.gz* | wc -l  # 应该有 14 个文件

# 检查文件完整性（如果有 MD5）
md5sum -c checksums.md5
```

### Q2: 解压后目录名不对
确保按照上面的重命名规则：
- `data` → `data_real`
- `data_syn_1` → 保持原名
- `models_cad` → 保持原名

### Q3: 磁盘空间不足
MP6D 数据集很大，建议：
- 只解压真实数据 (data.tar.gz) 用于测试：~14 GB
- 如果需要训练，再解压合成数据

---

## 🔗 参考

- **MP6D 数据集论文**: https://github.com/yhan9848/MP6D
- **FFB6D 原始代码**: https://github.com/ethnhe/FFB6D
- **DFTr 论文**: ICCV 2023

---

**配置完成后，运行 `./train_mp6d.sh` 开始训练！**
