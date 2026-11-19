import os
import glob
import shutil
import random

# =====================================================================
# 1. 路径与目标参数配置
# =====================================================================
SRC_DIR = r"E:\挑战sci\挑战sci\原始数据集\模拟数据8000+\he"
DST_DIR = r"F:\ultralyticsyolo8\datasets_Ue_middlesmall_target"
TOTAL_TARGET_COUNT = 600       # 目标总张数
SMALL_MEDIUM_RATIO = 0.8        # 80% 中小目标 (约 480 张)
MEDIUM_LARGE_RATIO = 0.2        # 20% 中大目标 (约 120 张)

def get_image_target_category(label_path):
    """
    解析 YOLO 格式标注文件 (.txt)，计算图像中所有目标框的平均面积占比
    返回分类: 'small_medium' (中小目标) 或 'medium_large' (中大目标)
    """
    if not os.path.exists(label_path):
        return None
    
    areas = []
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                w, h = float(parts[3]), float(parts[4])
                area = w * h
                areas.append(area)
                
    if not areas:
        return None
    
    # 计算图像中目标框的平均相对面积
    avg_area = sum(areas) / len(areas)
    
    # 判定阈值：相对面积 0.015 (相当于 640x640 图像下约 78x78 像素点)
    # 小于 0.015 归为中小目标，大于等于 0.015 归为中大目标
    if avg_area < 0.015:
        return 'small_medium'
    else:
        return 'medium_large'

def sample_and_copy():
    random.seed(42)  # 固定随机种子保证挑选结果可复现
    
    subsets = ['train', 'val', 'test']
    pool = {'small_medium': [], 'medium_large': []}
    
    print("🔍 正在扫描原始数据集样本...")
    for sub in subsets:
        img_dir = os.path.join(SRC_DIR, sub, 'images')
        lbl_dir = os.path.join(SRC_DIR, sub, 'labels')
        
        img_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.JPG', '*.PNG']:
            img_files.extend(glob.glob(os.path.join(img_dir, ext)))
            
        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base_name + '.txt')
            
            cat = get_image_target_category(lbl_path)
            if cat in pool:
                pool[cat].append((img_path, lbl_path))
                
    print(f"📊 扫描完成！获得候选池：")
    print(f"   - 中小目标候选池: {len(pool['small_medium'])} 张")
    print(f"   - 中大目标候选池: {len(pool['medium_large'])} 张")
    
    # 按照 8:2 比例计算抽样数量
    target_sm_count = int(TOTAL_TARGET_COUNT * SMALL_MEDIUM_RATIO)  # 480 张
    target_ml_count = TOTAL_TARGET_COUNT - target_sm_count         # 120 张
    
    sm_selected = random.sample(pool['small_medium'], min(target_sm_count, len(pool['small_medium'])))
    ml_selected = random.sample(pool['medium_large'], min(target_ml_count, len(pool['medium_large'])))
    
    all_selected = sm_selected + ml_selected
    random.shuffle(all_selected)
    
    print(f"🎯 选定样本：中小目标 {len(sm_selected)} 张，中大目标 {len(ml_selected)} 张，共 {len(all_selected)} 张。")
    
    # 按 8:1:1 比例重新划分为 train / val / test 目录结构
    num_total = len(all_selected)
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    
    train_set = all_selected[:num_train]
    val_set = all_selected[num_train:num_train + num_val]
    test_set = all_selected[num_train + num_val:]
    
    split_dict = {'train': train_set, 'val': val_set, 'test': test_set}
    
    print("🚀 正在组织并复制代码及标注文件至目标路径...")
    for split_name, samples in split_dict.items():
        dst_img_dir = os.path.join(DST_DIR, split_name, 'images')
        dst_lbl_dir = os.path.join(DST_DIR, split_name, 'labels')
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        for img_path, lbl_path in samples:
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(lbl_path)
            
            shutil.copy2(img_path, os.path.join(dst_img_dir, img_name))
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, lbl_name))
                
        print(f"   ✅ {split_name} 子集已构建完成: {len(samples)} 张")
        
    print(f"\n🎉 抽取并构建完成！新数据集已就绪：{DST_DIR}")

if __name__ == '__main__':
    sample_and_copy()