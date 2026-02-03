
# DFTr_Voting — Copilot 项目指引（复现用）

本仓库尝试复现论文《Deep Fusion Transformer Network with Weighted Vector-Wise Keypoints Voting for Robust 6D Object Pose Estimation》（ICCV 2023）。

本文档写给“在 VS Code 里由 Copilot/LLM 协助改代码”的场景：告诉你项目目标、代码地图、关键张量形状、训练/评估链路、以及最容易踩坑的地方。

---

## 0. 你正在复现什么（论文速记）

**核心思想**：
1.  **DFTr Block (Deep Fusion Transformer)**: 用 Transformer 替代简单的 concat 或 DenseFusion。在 RGB 分支（CNN）和 Point 分支（RandLA-Net）之间做双向 Cross-Attention，再拼起来做 Self-Attention，捕捉跨模态全局语义关联。
2.  **WVWV (Weighted Vector-Wise Voting)**: 既然 Point-wise offset 难学（尺度问题），不如学**指向关键点的单位向量场** + **权重（置信度）**。
    -   **解码**：无需 MeanShift 迭代，直接解一个加权最小二乘线性方程组 $Ax=b$ (Eq. 10-11)，速度快（109ms -> 18ms），精度高。

**关键公式**：
-   **Voting Objective**: $\hat{k}_j = \arg \min_{k_j} \sum_i c_i (P_i - k_j)^T (I - V_{i \to j} V_{i \to j}^T) (P_i - k_j)$
-   **Solution**: $k_j = A^{\dagger}b$，其中 $A = \sum c_i (I - V V^T)$， $b = \sum c_i (I - V V^T) P_i$。

**监督信号**：
-   Segmentation (Focal Loss)
-   Vector Field (L1 Loss, 预测单位向量)
-   Confidence/Weight (自监督/隐式学习，部分代码里看到用 log 辅助 loss)

---

## 1. 代码地图（从入口到输出）

| 模块 | 文件路径 | 职责 |
| :--- | :--- | :--- |
| **入口 (Train)** | `train_mp6d.py` | 训练主循环，数据加载，Loss 计算，模型保存。使用 `apex` 做混合精度训练。 |
| **入口 (Test)** | `train_mp6d.py` (with `-test`) | 测试入口，调用评估逻辑。实际脚本见 `test_mp6d.sh`。 |
| **模型主干** | `models/ffb6d_linemod.py` | `FFB6D` 类。包含 CNN (RGB) 和 RandLA-Net (Point) 分支。**关键修改点**：在 encoder/decoder 各层插入了 `DFTr` 模块。 |
| **Fusion 模块** | `models/my_fusion_block/DFTr.py` | `DFTr` 类。实现 Transformer 融合 (CrossAttn + SelfAttn + PosEmbed)。 |
| **Voting 算法** | `models/utils_my/iteration_decode_kps.py` | `vector2Kps` 类。核心函数 `linear_decode_point` 实现了 WVWV 线性方程求解。 |
| **评估流程** | `models/utils_my/pvn3d_eval_utils_kpls_v1.py` | `cal_frame_poses`：从 mask -> 2D/3D 点 -> 调用 `vector2Kps` -> 得到 Keypoints -> LeastSquares 算 Pose。 |
| **Loss** | `models/loss.py` | `OFLoss` (Vector Field L1), `FocalLoss`, `CosLoss` (Cosine Similarity)。 |
| **Dataset** | `datasets/MP6D/MP6D_dataset_ori.py` | MP6D 数据读取。负责生成 GT：`kp_targ_ofst` (关键点指向向量), `ctr_targ_ofst` (中心点指向向量)。 |

注意：项目根目录文件较乱（有带空格的文件名），尽量引用 `models/` 下的结构化代码。

---

## 2. 数据与标注：网络学的到底是什么

**关键张量 (Tensor)**:
-   `kp_targ_ofst`: Shape `(BS, N_pts, N_kps, 3)`.
    -   **含义**: 点云中每个点 $P_i$ 指向关键点 $K_j$ 的偏移量 ($P_i - K_j$ 还是 $K_j - P_i$ 需看代码实现，MP6D loader 里是 `cld - kp`)。
    -   **单位**: 米 (Meters)。`MP6D_dataset_ori.py` 中 `t = t / 1000.0`。
    -   **归一化**: **注意**，虽然论文提倡学习 Unit Vector，但现行代码在 Dataset 和 Training 中使用的是 **Raw Offsets (未归一化)**。
        -   训练时用 L1 Loss 回归真实偏移。
        -   测试时 (`pvn3d_eval_utils_kpls_v1.py`) 会强制 `torch.div(..., norm)` 转为单位向量给 Voting 算法用。
-   `ctr_targ_ofst`: Shape `(BS, N_pts, 3)`. 指向物体中心的偏移量。
-   `labels`: Shape `(BS, N_pts, 1)`. 语义分割标签。

**Tips**: 如果你要改 Loss 强制学 Unit Vector，记得同步放开 `Cosine Loss` 并修改 Dataset 的归一化注释。

---

## 3. 模型结构：FFB6D + DFTr 插入位置

**骨干网络**:
-   **RGB Branch**: ResNet34 + PSPNet (提取多尺度 CNN 特征)。
-   **Point Branch**: RandLA-Net (提取点云几何特征)。

**DFTr (Deep Fusion Transformer)**:
-   **位置**: 替代了原 FFB6D 的简单的 `DenseFusion` 模块，插入在 Encoder 和 Decoder 的每一层跳跃连接处。
-   **输入**: RGB Feature map (`B, C, H*W`) 和 Point Features (`B, C, N`).
-   **流程**:
    1.  **Bi-Directional Cross-Attention**: RGB query Point, Point query RGB.
    2.  **Concatenation**: 拼成长序列 `(B, H*W + N, C)`.
    3.  **Positional Embedding**: 加上可学习的 embedding。
    4.  **Self-Attention Block**: 标准 Transformer Encoder 层。
    5.  **Split**: 拆回 RGB 和 Point 特征，送回各自的主干网络。

**输出头 (Heads)**:
-   `pred_rgbd_segs`: `(BS, N_pts, n_cls)` - 分割 logits。
-   `pred_kp_ofs`: `(BS, n_kps, N_pts, 3)` - 关键点偏移。
-   `pred_kp_ofs_score`: `(BS, n_kps, N_pts, 1)` - **Keypoint Voting Confidence**. (Sigmoid 激活)。
    -   这是 WVWV 算法中 $c_i$ 的来源，表示该点对关键点预测的置信度。

---

## 4. WVWV：Weighted Vector-Wise Voting（从公式到代码）

**论文原理**:
-   每个点 $P_i$ 预测一个指向关键点 $K_j$ 的单位向量 $V_{i \to j}$。
-   理想情况下，$P_i + s \cdot V_{i \to j} = K_j$（$s$是距离），消去 $s$ 可得几何约束：$K_j$ 必须在以 $P_i$ 为起点、方向为 $V_{i \to j}$ 的射线上。
-   数学表达：$(P_i - K_j)$ 与 $V_{i \to j}$ 平行 $\Rightarrow (P_i - K_j) \times V_{i \to j} = 0$。
-   或者是最小化投影误差：$(I - V V^T)(P_i - K_j) = 0$。

**代码实现 (`iteration_decode_kps.py`)**:
1.  **Top-K 筛选**: 根据 `pred_kp_ofs_score` 选出置信度最高的 K 个点（K=80 或 K=Points）。
2.  **构建方程**:
    -   $A = \sum_{i \in TopK} c_i (I - V_i V_i^T)$
    -   $b = \sum_{i \in TopK} c_i (I - V_i V_i^T) P_i$
3.  **求解**:
    -   `Kp = np.linalg.lstsq(R[kps_id], q[kps_id], rcond=None)[0].T`
    -   代码中变量对应：`R` -> $A$, `q` -> $b$。

**优势**:
-   相比 MeanShift (迭代聚类)，这是解析解 (Closed-form solution)，一次计算即可，极其高效。

---

## 5. Loss 与训练策略

**总 Loss**: $L = \lambda_1 L_{seg} + \lambda_2 L_{vecf}$

1.  **Segmentation Loss ($L_{seg}$)**:
    -   使用 `FocalLoss` (`gamma=2, alpha=0.5`) 解决前景背景不平衡。

2.  **Vector Field Loss ($L_{vecf}$)**:
    -   函数: `of_l1_loss` in `models/loss.py`。
    -   **关键机制 (Uncertainty Learning)**:
        $$L = |Offset_{pred} - Offset_{gt}| \cdot s - w \cdot \log(s)$$
        -   $s$ (`pred_ofsts_score`): 网络预测的置信度。
        -   **直观理解**: 如果预测偏差大，网络会自动降低 $s$ 来减少第一项 Loss；但第二项 `-log(s)` 惩罚过小的 $s$，防止全 0 坍缩。
    -   **注意**: 代码中有 `Cosine Loss` 但默认被注释掉了 (`model_fn_decorator` 中)。目前主要靠 L1 回归。

**超参数**:
-   `w` (log项系数): 0.01
-   `batch_size`: 3 (config 中) -> 显存占用较大。
-   `lr`: 1e-2, Cyclical Learning Rate.

---

## 6. 运行方式（训练/测试/常见调试）

**训练**:
```bash
./train_mp6d.sh
# 核心命令
python3 -m torch.distributed.launch --nproc_per_node=2 train_mp6d.py --gpus=2 --gpu='0,1'
```

**测试**:
```bash
./test_mp6d.sh
# 核心命令
python3 train_mp6d.py -eval_net -checkpoint <PATH> -test -test_pose
```

**调试 Tip**:
-   只想跑通流程? 加 `-debug` 参数。
-   想看 Tensorboard? 运行 `tensorboard --logdir train_log/MP6D/train_info`.

---

## 7. 已知坑与修复建议（很重要）

1.  **文件路径地狱**:
    -   代码里硬编码了大量 `/home/rubbish/jun/...` 这种绝对路径。
    -   **修复**: 全局搜索 `/home/rubbish`，替换为 `config.log_dir` 或相对路径。
    -   `test_mp6d.sh` 里的 checkpoint 路径通常是错的，记得改。

2.  **文件名空格**:
    -   `models/utils_my/iteration_decode_kps.py` 可能原本叫 `iteration_decode_kps .py`? VS Code 里如果看到带空格的文件，建议重命名并更新 import。
    -   `models/utils_my/__init__ .py` 也是同理。

3.  **NaN 问题**:
    -   WVWV 解方程 `lstsq` 当 $A$ 奇异时（例如所有向量共线）会崩或出 NaN。
    -   `models/loss.py` 里 `torch.log(prob)` 需要 clamp 避免 log(0)。代码目前有 `clamp(min=1e-7)`，是安全的。

---

## 8. 你要改什么时，应该怎么改（Copilot 工作约束）

1.  **小步快跑**: 先改路径/配置跑通 `-debug`，再上全量数据。
2.  **保持接口**: `DFTr` 模块的输入输出 `(rgb_feat, point_feat)` 尽量不动，魔改内部即可。
3.  **验证**: 改完模型结构后，务必检查 `model parameters` 打印量是否符合预期（~175M params）。


