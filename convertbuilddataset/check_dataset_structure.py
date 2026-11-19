import os
import glob
import cv2
import numpy as np

def check_yolo_dataset(dataset_root):
    print(f"================ 🔍 正在探测数据集目录 ================")
    print(f"📁 根目录路径: {dataset_root}")
    
    if not os.path.exists(dataset_root):
        print(f"❌ 错误: 路径 {dataset_root} 不存在，请检查文件夹名称或路径是否正确！")
        return

    print(f"📂 根目录子项: {os.listdir(dataset_root)}")
    
    # 检查 train 和 val 目录
    splits = ['train', 'val']
    for split in splits:
        print(f"\n---------------- 📦 检查 [{split.upper()}] 子集 ----------------")
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            print(f"⚠️ 警告: 未找到 {split} 文件夹！")
            continue
            
        img_dir = os.path.join(split_dir, 'images')
        label_dir = os.path.join(split_dir, 'labels')
        
        has_img_dir = os.path.exists(img_dir)
        has_label_dir = os.path.exists(label_dir)
        
        print(f"  - images 目录存在: {has_img_dir} ({img_dir})")
        print(f"  - labels 目录存在: {has_label_dir} ({label_dir})")
        
        if not has_img_dir or not has_label_dir:
            print(f"  ❌ {split} 目录结构不完整（缺少 images 或 labels）")
            continue
            
        # 统计图片与标签文件
        img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        img_list = []
        for ext in img_extensions:
            img_list.extend(glob.glob(os.path.join(img_dir, ext)))
            
        txt_list = glob.glob(os.path.join(label_dir, '*.txt'))
        
        print(f"  - 图片总数: {len(img_list)} 张")
        print(f"  - 标签文件数: {len(txt_list)} 个")
        
        # 配对检查与异常排查
        missing_labels = 0
        empty_labels = 0
        total_instances = 0
        class_counts = {}
        
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            base_name, _ = os.path.splitext(img_name)
            txt_path = os.path.join(label_dir, base_name + '.txt')
            
            if not os.path.exists(txt_path):
                missing_labels += 1
            else:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        empty_labels += 1
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            total_instances += 1
                            cls_id = int(parts[0])
                            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                            
        print(f"  - ⚠️ 缺少对应标签的图片数: {missing_labels} 张")
        print(f"  - ⚠️ 内容为空的标签文件数: {empty_labels} 个")
        print(f"  - 📈 总标注实例数 (Instances): {total_instances}")
        print(f"  - 🏷️ 各类别实例分布: {class_counts}")

    # 检查是否有 yaml 配置文件
    yaml_files = glob.glob(os.path.join(dataset_root, '*.yaml'))
    print(f"\n---------------- 📄 YAML 配置文件检查 ----------------")
    if len(yaml_files) > 0:
        print(f"✅ 发现配置文件: {[os.path.basename(y) for y in yaml_files]}")
    else:
        print(f"⚠️ 提示: 根目录下未发现 .yaml 配置文件，建议补一个配对的 dataset.yaml。")
    print("=" * 60)

if __name__ == '__main__':
    # 目标探测路径
    target_dataset_path = r"F:\ultralyticsyolo8\datasets_stage3_small__D_fire"
    check_yolo_dataset(target_dataset_path)