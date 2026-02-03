#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证 Linemod 数据集和环境配置
Usage: python3 verify_setup.py
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

def check_mark(condition, msg):
    """打印检查结果"""
    if condition:
        print(f"✅ {msg}")
        return True
    else:
        print(f"❌ {msg}")
        return False

def main():
    print("\n" + "="*80)
    print("DFTr_Voting - Linemod 数据集环境验证")
    print("="*80 + "\n")
    
    project_root = Path(__file__).parent
    all_good = True
    
    # 1. 检查数据集目录
    print("📁 检查数据集目录...")
    linemod_root = project_root / "datasets/linemod/Linemod_preprocessed"
    all_good &= check_mark(linemod_root.exists(), 
                          f"Linemod 数据集根目录: {linemod_root}")
    
    data_dir = linemod_root / "data"
    all_good &= check_mark(data_dir.exists(), 
                          f"数据目录: {data_dir}")
    
    models_dir = linemod_root / "models"
    all_good &= check_mark(models_dir.exists(), 
                          f"模型目录: {models_dir}")
    
    # 2. 检查物体目录（13个物体）
    print("\n🎯 检查物体数据...")
    obj_ids = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
    obj_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 
                 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
    
    obj_count = 0
    for obj_id, obj_name in zip(obj_ids, obj_names):
        obj_dir = data_dir / obj_id
        if obj_dir.exists():
            # 检查必要的子目录
            rgb_dir = obj_dir / "rgb"
            depth_dir = obj_dir / "depth"
            mask_dir = obj_dir / "mask"
            gt_file = obj_dir / "gt.yml"
            
            if all([rgb_dir.exists(), depth_dir.exists(), mask_dir.exists(), gt_file.exists()]):
                obj_count += 1
                print(f"  ✅ {obj_id}: {obj_name:12s} - RGB/Depth/Mask/GT 完整")
            else:
                print(f"  ⚠️  {obj_id}: {obj_name:12s} - 部分文件缺失")
        else:
            print(f"  ❌ {obj_id}: {obj_name:12s} - 目录不存在")
    
    all_good &= check_mark(obj_count == 13, f"找到 {obj_count}/13 个物体")
    
    # 3. 检查 3D 模型文件
    print("\n🎨 检查 3D 模型...")
    model_count = 0
    for obj_id in obj_ids:
        model_file = models_dir / f"obj_{obj_id}.ply"
        if model_file.exists():
            model_count += 1
    
    all_good &= check_mark(model_count == 13, f"找到 {model_count}/13 个 PLY 模型")
    
    # 检查 models_info.yml
    models_info = models_dir / "models_info.yml"
    if models_info.exists():
        with open(models_info, 'r') as f:
            info = yaml.safe_load(f)
        check_mark(True, f"models_info.yml 包含 {len(info)} 个物体信息")
    else:
        check_mark(False, "models_info.yml 文件不存在")
    
    # 4. 检查 Python 依赖
    print("\n📦 检查 Python 依赖...")
    try:
        import torch
        check_mark(True, f"PyTorch {torch.__version__}")
    except ImportError:
        check_mark(False, "PyTorch 未安装")
        all_good = False
    
    try:
        import cv2
        check_mark(True, f"OpenCV {cv2.__version__}")
    except ImportError:
        check_mark(False, "OpenCV 未安装")
        all_good = False
    
    try:
        from apex import amp
        check_mark(True, "NVIDIA Apex (混合精度训练)")
    except ImportError:
        check_mark(False, "NVIDIA Apex 未安装（可选，用于加速训练）")
    
    # 5. 检查关键点目录（可能需要后续生成）
    print("\n🔑 检查关键点配置...")
    kps_dir = project_root / "datasets/linemod/lm_obj_kps"
    check_mark(kps_dir.exists(), f"关键点目录: {kps_dir}")
    
    # 6. 检查模型代码
    print("\n🧠 检查模型代码...")
    ffb6d_file = project_root / "models/ffb6d_linemod.py"
    all_good &= check_mark(ffb6d_file.exists(), "FFB6D 模型文件")
    
    dftr_file = project_root / "models/my_fusion_block/DFTr.py"
    all_good &= check_mark(dftr_file.exists(), "DFTr 融合模块")
    
    voting_file = project_root / "models/utils_my/iteration_decode_kps.py"
    all_good &= check_mark(voting_file.exists(), "WVWV 投票算法")
    
    # 7. 检查训练脚本
    print("\n🚀 检查训练脚本...")
    train_script = project_root / "train_linemod.py"
    check_mark(train_script.exists(), "train_linemod.py")
    
    test_script = project_root / "test_linemod.sh"
    check_mark(test_script.exists(), "test_linemod.sh")
    
    # 总结
    print("\n" + "="*80)
    if all_good:
        print("🎉 恭喜！环境验证通过，可以开始训练了！")
        print("\n下一步：")
        print("  1. 生成关键点文件（如果需要）：")
        print("     python3 generate_keypoints.py")
        print("\n  2. 运行 debug 模式验证流程：")
        print("     python3 train_linemod.py --cls=ape --gpu='0' -debug")
        print("\n  3. 开始正式训练：")
        print("     ./train_linemod.sh ape")
    else:
        print("⚠️  发现一些问题，请根据上面的提示修复后再继续。")
        print("\n常见问题：")
        print("  - 数据集未解压：unzip Linemod_preprocessed.zip -d datasets/linemod/")
        print("  - 缺少依赖：pip install torch opencv-python pyyaml")
    print("="*80 + "\n")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
