import os
import shutil
import glob
import random
import numpy as np
import cv2

def safe_imread(img_path):
    """完美解决 Windows 中文路径/乱码读取问题"""
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"⚠️ 读取图片异常 {img_path}: {e}")
        return None

def filter_pure_fire_dataset(source_train_img_dir, source_train_label_dir, output_dataset_dir, target_count=500, val_ratio=0.2):
    """
    1. 仅保留【纯火焰】图片：
       - 图片中必须包含火焰（class 0 或对应索引）。
       - 绝对不能包含烟雾或其他杂类（如果检测到非火焰类别直接剔除）。
       - 采用 COCO 标准：小目标面积 < 32*32 且中大目标数量占比 <= 10%。
    2. 自动收集目标数量（如 500 张）。
    3. 自动按 8:2 划分为 train / val，并重写标签确保类别正确归零，生成 YAML。
    """
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dataset_dir, split, 'labels'), exist_ok=True)
        
    img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_list = []
    for ext in img_extensions:
        img_list.extend(glob.glob(os.path.join(source_train_img_dir, ext)))
        
    print(f"🔍 正在扫描源目录: {source_train_img_dir}")
    print(f"📦 发现总图片数: {len(img_list)} 张，开始严苛筛选【纯火焰、纯小目标】图片...")
    
    valid_samples = []
    
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_name)
        txt_path = os.path.join(source_train_label_dir, base_name + ".txt")
        
        if not os.path.exists(txt_path):
            continue
            
        img = safe_imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        fire_instances = 0
        other_instances = 0
        small_fire_count = 0
        medium_large_fire_count = 0
        
        cleaned_lines = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # 假设在原 D-Fire 中 0 代表火，如果你的数据集中火焰是其他 ID，请在此调整
                    # 这里严格限定：如果出现非 0 类别（如烟雾），则直接判为包含杂质，整张图抛弃！
                    if cls_id != 0:
                        other_instances += 1
                        continue
                        
                    fire_instances += 1
                    bw_norm = float(parts[3])
                    bh_norm = float(parts[4])
                    
                    abs_w = bw_norm * w
                    abs_h = bh_norm * h
                    area = abs_w * abs_h
                    
                    # 统计小目标比例
                    if area < 32 * 32:
                        small_fire_count += 1
                    else:
                        medium_large_fire_count += 1
                        
                    # 重写为单类 0 (fire)
                    cleaned_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                    
        # 严格筛选条件：
        # 1. 必须有火焰 (fire_instances > 0)
        # 2. 绝对不能有烟雾或其他杂类 (other_instances == 0)
        # 3. 必须包含小目标 (small_fire_count > 0)
        # 4. 中大目标数量占比不超过 10%
        if fire_instances > 0 and other_instances == 0 and small_fire_count > 0:
            total = fire_instances
            ratio = medium_large_fire_count / total
            if ratio <= 0.1:
                valid_samples.append((img_path, txt_path, img_name, base_name, cleaned_lines))
                
    print(f"✨ 符合【纯火焰、纯小目标、无烟雾】条件的图片共有: {len(valid_samples)} 张")
    
    random.seed(42)
    random.shuffle(valid_samples)
    
    if len(valid_samples) > target_count:
        valid_samples = valid_samples[:target_count]
        print(f"✂️ 已截取前 {target_count} 张高质量纯火焰图片。")
    else:
        print(f"⚠️ 符合条件的图片总数（{len(valid_samples)}张）未达到设定的 {target_count} 张，将全部采用。")
        
    # 划分 Train 和 Val (8:2)
    val_size = int(len(valid_samples) * val_ratio)
    val_samples = valid_samples[:val_size]
    train_samples = valid_samples[val_size:]
    
    print(f"📊 数据集划分完成 -> 训练集 (Train): {len(train_samples)} 张 | 验证集 (Val): {len(val_samples)} 张")
    
    # 写入文件与清洗后的标签
    for sample_list, split_name in [(train_samples, 'train'), (val_samples, 'val')]:
        for img_p, txt_p, img_n, base_n, clean_lines in sample_list:
            dest_img = os.path.join(output_dataset_dir, split_name, 'images', img_n)
            dest_txt = os.path.join(output_dataset_dir, split_name, 'labels', base_n + '.txt')
            
            shutil.copy(img_p, dest_img)
            with open(dest_txt, 'w', encoding='utf-8') as f_out:
                f_out.writelines(clean_lines)
                
    # 自动生成单类纯火焰 YAML
    yaml_content = f"""path: {output_dataset_dir.replace(os.sep, '/')}
train: train/images
val: val/images

nc: 1
names:
  0: fire
"""
    yaml_path = os.path.join(output_dataset_dir, "pure_fire.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"🎉 纯火焰数据集制作完成！YAML 配置文件已生成: {yaml_path}")

if __name__ == '__main__':
    train_img_dir = r"E:\挑战sci\挑战sci\原始数据集\模拟数据8000+\he\train\images"
    train_label_dir = r"E:\挑战sci\挑战sci\原始数据集\模拟数据8000+\he\train\labels"
    output_target_dir = r"F:\ultralyticsyolo8\datasets_D-fire-pure-small"
    
    filter_pure_fire_dataset(train_img_dir, train_label_dir, output_target_dir, target_count=500, val_ratio=0.2)