import os
import glob
import shutil
import random
from collections import defaultdict

# 1. 路径设置
src_dataset_dir = r"E:\挑战sci\挑战sci\原始数据集\D-fire"
dst_dataset_dir =r"F:\ultralyticsyolo8\datasets_stage4_micro_fire_only"

# 2. 参数设置
FIRE_CLASS_ID = 0              # 原 D-fire 数据集中 fire 的类别索引
SMALL_THRESH = (32 / 640) ** 2  # COCO 标准小目标相对面积阈值 0.0025
SPLIT_RATIO = [0.8, 0.1, 0.1]   # Train : Val : Test 划分比例
SEED = 42

random.seed(SEED)

def process_and_extract_micro_fire():
    print(f"================ 🚀 开始从原始 D-fire 中提取【纯小火焰】样本 ================")
    print(f"📂 原始数据集路径: {src_dataset_dir}")
    print(f"📁 目标生成路径  : {dst_dataset_dir}\n")

    # 检索原始数据中的所有标注文件 (.txt)
    all_txt_files = glob.glob(os.path.join(src_dataset_dir, "**", "*.txt"), recursive=True)
    
    valid_samples = []  # 存放包含“小火焰”框的图像与标注路径对
    total_raw_images = 0
    total_fire_boxes = 0
    small_fire_boxes = 0

    for txt_path in all_txt_files:
        # 匹配相对应的图片路径 (支持 .jpg, .png, .jpeg)
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            candidate = txt_path.rsplit('.', 1)[0] + ext
            # 兼容 labels 和 images 分开存储或存储在同一目录的形态
            if not os.path.exists(candidate):
                candidate = candidate.replace(f"{os.sep}labels{os.sep}", f"{os.sep}images{os.sep}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            continue

        total_raw_images += 1

        # 读取并过滤标签
        valid_lines = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            area = w * h

            # 🎯 条件：仅保留 fire 且 相对面积 < 0.0025 的微小目标
            if cls_id == FIRE_CLASS_ID:
                total_fire_boxes += 1
                if area < SMALL_THRESH:
                    small_fire_boxes += 1
                    # 重置类别索引为 0 (单类别 fire)
                    valid_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")

        # 只要包含至少 1 个小火焰框，即视为有效样本
        if len(valid_lines) > 0:
            valid_samples.append((img_path, valid_lines))

    print(f"📊 原始图像扫描完成：")
    print(f"   - 原始图像总量: {total_raw_images} 张")
    print(f"   - 原始火焰总框数: {total_fire_boxes} 个")
    print(f"   - 筛选出的【微小火焰框】总量: {small_fire_boxes} 个")
    print(f"   - 包含微小火焰的【有效图像总数】: {len(valid_samples)} 张\n")

    if len(valid_samples) == 0:
        print("❌ 未搜索到满足条件的小火焰样本，请检查原始数据集路径或阈值设置！")
        return

    # 打乱样本并按 8:1:1 切分
    random.shuffle(valid_samples)
    n_total = len(valid_samples)
    n_train = int(n_total * SPLIT_RATIO[0])
    n_val = int(n_total * SPLIT_RATIO[1])
    
    splits = {
        'train': valid_samples[:n_train],
        'val': valid_samples[n_train:n_train + n_val],
        'test': valid_samples[n_train + n_val:]
    }

    # 创建目标目录结构并复制文件
    for split_name, samples in splits.items():
        img_dst_dir = os.path.join(dst_dataset_dir, split_name, "images")
        lbl_dst_dir = os.path.join(dst_dataset_dir, split_name, "labels")
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(lbl_dst_dir, exist_ok=True)

        box_count = 0
        for img_path, lines in samples:
            base_name = os.path.basename(img_path)
            txt_name = os.path.splitext(base_name)[0] + ".txt"

            # 复制图片
            shutil.copy(img_path, os.path.join(img_dst_dir, base_name))
            
            # 写入清洗后的微小火焰标签
            with open(os.path.join(lbl_dst_dir, txt_name), 'w') as f:
                f.writelines(lines)
            
            box_count += len(lines)

        print(f"✅ [{split_name.upper()} 集] 生成完毕:")
        print(f"   - 图像数量: {len(samples)} 张")
        print(f"   - 小火焰目标框数量: {box_count} 个")

    # 自动生成对应的 dataset.yaml 文件
    yaml_content = f"""# Pure Micro-Fire Dataset (Class 0: fire)
path: {dst_dataset_dir}
train: train/images
val: val/images
test: test/images

names:
  0: fire
"""
    yaml_path = os.path.join(dst_dataset_dir, "micro_fire_only.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n🎉 专属【微小火焰】数据集构建完成！配置文件已生成至:")
    print(f"📄 {yaml_path}")

if __name__ == '__main__':
    process_and_extract_micro_fire()