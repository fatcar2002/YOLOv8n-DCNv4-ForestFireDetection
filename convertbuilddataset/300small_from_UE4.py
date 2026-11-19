import os
import shutil
import random
from pathlib import Path

# =====================================================================
# 路径配置
# =====================================================================
SRC_ROOT = Path(r"E:\挑战sci\挑战sci\原始数据集\模拟数据8000+\he")
DST_ROOT = Path(r"F:\ultralyticsyolo8\datasets300test_small")

# 抽样数量配置
NUM_TRAIN = 240   # 训练集：240 张
NUM_VAL = 30      # 验证集：30 张
NUM_TEST = 30     # 测试集：30 张

# 小目标面积阈值（框面积占整图比例小于 1%，即 0.01）
SMALL_TARGET_AREA_THRESH = 0.01 

def is_small_target_label(label_file):
    """读取 txt 标注文件，判断是否包含小目标"""
    if not label_file.exists():
        return False
    
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # YOLO 格式: class x_center y_center width height
            w = float(parts[3])
            h = float(parts[4])
            area = w * h  # 相对面积 (0~1之间)
            
            # 如果存在相对面积小于 1% 的目标，判定为小目标样本
            if area <= SMALL_TARGET_AREA_THRESH:
                return True
    return False

def collect_small_target_samples(split_name):
    """从源数据集对应的 split 中搜集小目标样本 (img_path, label_path)"""
    img_dir = SRC_ROOT / split_name / "images"
    lbl_dir = SRC_ROOT / split_name / "labels"
    
    # 支持常见的图片后缀
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts]
    
    matched_pairs = []
    for img_path in img_files:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if is_small_target_label(lbl_path):
            matched_pairs.append((img_path, lbl_path))
            
    return matched_pairs

def build_mini_dataset():
    print("================ 🚀 启动小目标 Mini 数据集筛选与构建 ================")
    
    # 1. 扫描 train, val, test 中的所有小目标样本
    all_pairs = []
    for split in ["train", "val", "test"]:
        pairs = collect_small_target_samples(split)
        print(f"🔍 从原始 {split} 集中筛选出小目标样本: {len(pairs)} 张")
        all_pairs.extend(pairs)
        
    total_needed = NUM_TRAIN + NUM_VAL + NUM_TEST
    print(f"📊 筛选到的总小目标池大小: {len(all_pairs)} 张 | 需要抽取: {total_needed} 张")
    
    if len(all_pairs) < total_needed:
        print(f"⚠️ 警告：符合严格小目标条件的样本不足 {total_needed} 张，将随机打乱全部抽取。")
        selected_pairs = all_pairs
    else:
        # 打乱随机抽样，保证涵盖不同场景
        random.seed(42)  # 固定随机种子，保证可复现
        selected_pairs = random.sample(all_pairs, total_needed)
        
    # 2. 划分选出的数据集
    train_pairs = selected_pairs[:NUM_TRAIN]
    val_pairs = selected_pairs[NUM_TRAIN:NUM_TRAIN + NUM_VAL]
    test_pairs = selected_pairs[NUM_TRAIN + NUM_VAL:]
    
    splits_map = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }
    
    # 3. 复制文件并建立目录
    for split_name, pairs in splits_map.items():
        dst_img_dir = DST_ROOT / split_name / "images"
        dst_lbl_dir = DST_ROOT / split_name / "labels"
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_p, lbl_p in pairs:
            shutil.copy(img_p, dst_img_dir / img_p.name)
            shutil.copy(lbl_p, dst_lbl_dir / lbl_p.name)
            
        print(f"✅ 已完成 {split_name} 集构建: {len(pairs)} 张图像及标注文件")
        
    # 4. 自动生成专属 dataset.yaml 配置文件
    yaml_content = f"""path: {DST_ROOT}
train: train/images
val: val/images
test: test/images

names:
  0: fire
  1: smoke
"""
    yaml_path = DST_ROOT / "stage3_mini300_dfire.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"\n🎉 专属 Mini 数据集 YAML 配置文件已自动生成至:")
    print(f"📄 {yaml_path}")
    print("=====================================================================")

if __name__ == "__main__":
    build_mini_dataset()