# Linemod 数据集复现指南

## 📦 步骤 1: 解压数据集

你已经下载了 `Linemod_preprocessed.zip`，现在需要解压：

```bash
# 在项目根目录执行
cd /home/lixinlong/Project/DFTr_Voting

# 创建数据集目录
mkdir -p datasets/linemod

# 解压数据集（大约 8.4GB）
unzip Linemod_preprocessed.zip -d datasets/linemod/

# 检查解压结果
ls -la datasets/linemod/Linemod_preprocessed/
```

预期目录结构：
```
datasets/linemod/
└── Linemod_preprocessed/
    ├── data/         # RGB-D 图像和标注
    ├── models/       # 3D 物体模型
    └── ...
```

---

## 🔧 步骤 2: 创建 Linemod 数据集加载器

项目目前只有 `MP6D_dataset_ori.py`，需要创建对应的 Linemod 版本：

```bash
# 方法1：复制 MP6D 的实现并修改
cp datasets/MP6D/MP6D_dataset_ori.py datasets/linemod/linemod_dataset.py

# 方法2：参考 FFB6D 原仓库的实现
# https://github.com/ethnhe/FFB6D/blob/master/ffb6d/datasets/linemod/linemod_dataset.py
```

**关键修改点**：
- 数据路径指向 `datasets/linemod/Linemod_preprocessed/`
- 类别列表：`['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']`（共13个物体）
- Linemod 是**单物体训练**（与 MP6D 的多物体场景不同）

---

## 🚀 步骤 3: 准备关键点文件

Linemod 需要每个物体的关键点定义（用于 WVWV 投票）：

```bash
mkdir -p datasets/linemod/lm_obj_kps
mkdir -p datasets/linemod/kps_orb9_fps
```

关键点文件格式（参考 `datasets/MP6D/MP6D_kps/`）：
- `ape_8_kps.txt` - 8个关键点坐标
- `ape_corners.txt` - 3D bbox 角点
- `ape_fps.txt` - FPS采样点

**获取方式**：
1. 从 FFB6D 原仓库下载：https://github.com/ethnhe/FFB6D
2. 或使用 `models/utils_my/basic_utils.py` 中的 `get_kps()` 函数自动生成

---

## 📊 步骤 4: 配置检查

确认 `common.py` 中 Linemod 的配置正确：

```python
# common.py 第130-176行已有配置
self.dataset_name == 'linemod'
self.lm_root = 'datasets/linemod/'  # 确保路径正确
```

---

## 🏋️ 步骤 5: 训练模型

### **单物体训练示例（推荐从 ape 开始）**

```bash
# 给脚本添加执行权限
chmod +x train_linemod.sh test_linemod.sh

# 训练 ape 物体（双GPU）
./train_linemod.sh ape

# 或单GPU训练
python3 train_linemod.py --cls=ape --gpu='0'

# Debug 模式（快速验证流程）
python3 train_linemod.py --cls=ape --gpu='0' -debug
```

### **训练其他物体**
```bash
./train_linemod.sh cat
./train_linemod.sh can
# ... 依次训练13个物体
```

---

## 🧪 步骤 6: 测试评估

```bash
# 测试单个物体
./test_linemod.sh ape train_log/linemod/checkpoints/ape_best.pth.tar

# 或直接用命令
python3 train_linemod.py \
    --cls=ape \
    -eval_net \
    -checkpoint train_log/linemod/checkpoints/ape_best.pth.tar \
    -test -test_pose
```

---

## ⚠️ 常见问题

### 问题1：找不到数据集模块
```
ImportError: No module named 'datasets.linemod.linemod_dataset'
```

**解决**：需要创建 `datasets/linemod/linemod_dataset.py`。可以：
- 复制 `datasets/MP6D/MP6D_dataset_ori.py` 并修改路径
- 或从 FFB6D 原仓库获取

### 问题2：路径错误
代码中硬编码了 `/home/rubbish/jun/...`，需要全局替换为你的路径。

```bash
# 搜索并替换（在 VS Code 中或用命令）
grep -r "/home/rubbish" . --exclude-dir=__pycache__
```

### 问题3：关键点文件缺失
```
FileNotFoundError: datasets/linemod/lm_obj_kps/ape_8_kps.txt
```

**解决**：从 FFB6D 下载或使用 `basic_utils.py` 生成。

### 问题4：显存不足
Linemod 的 batch_size 默认较大，可以修改：

```python
# common.py 中
self.mini_batch_size = 1  # 改小一点
```

---

## 📈 预期结果

论文在 Linemod 数据集上的性能（ADD(-S) metric）：
- 平均精度：~90%+ 
- 推理速度：~18ms/帧（WVWV 解码）

训练日志位置：
- Tensorboard: `train_log/linemod/train_info/`
- Checkpoints: `train_log/linemod/checkpoints/`

查看训练曲线：
```bash
tensorboard --logdir train_log/linemod/train_info
```

---

## 🔗 参考资源

1. **FFB6D 原仓库**（数据集参考）：https://github.com/ethnhe/FFB6D
2. **论文**：Deep Fusion Transformer Network (ICCV 2023)
3. **MP6D 数据集**：https://github.com/yhan9848/MP6D

---

## 🎯 快速开始（一键命令）

```bash
# 1. 解压数据集
unzip Linemod_preprocessed.zip -d datasets/linemod/

# 2. 运行 debug 模式验证环境
python3 train_linemod.py --cls=ape --gpu='0' -debug

# 3. 如果报错缺少 linemod_dataset.py，请参考上面"步骤2"创建
```

---

**祝复现顺利！如有问题，可以参考项目根目录的 `.github/copilot-instructions.md`。**
