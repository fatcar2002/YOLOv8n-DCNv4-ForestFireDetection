import os
import glob
import cv2
from collections import defaultdict

def analyze_yolo_dataset(dataset_root, dataset_name):
    print(f"\n================ 📊 正在统计数据集: {dataset_name} ================")
    print(f"📁 路径: {dataset_root}")
    
    if not os.path.exists(dataset_root):
        print(f"⚠️ 警告: 路径 {dataset_root} 不存在，请检查挂载盘符或文件夹名称。")
        return

    # 统计容器
    stats = {
        "splits": {},
        "total_images": 0,
        "total_instances": 0,
        "small": 0,  # 面积 < 32^2 (1024)
        "medium": 0, # 32^2 <= 面积 < 96^2 (9216)
        "large": 0   # 面积 >= 96^2
    }

    # 遍历常见的 splits (train, val, test)
    splits = ["train", "val", "test", "images"]
    for split in splits:
        img_dir = os.path.join(dataset_root, split, "images")
        if not os.path.exists(img_dir):
            # 尝试直接是 split 目录
            img_dir = os.path.join(dataset_root, split)
        
        if os.path.exists(img_dir) and os.path.isdir(img_dir):
            img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            img_list = []
            for ext in img_extensions:
                img_list.extend(glob.glob(os.path.join(img_dir, ext)))
            
            split_img_count = len(img_list)
            split_instances = 0
            
            if split_img_count > 0:
                stats["splits"][split] = {"images": split_img_count, "instances": 0}
                stats["total_images"] += split_img_count
                
                # 寻找对应的 labels 目录
                label_dir = img_dir.replace("images", "labels")
                if not os.path.exists(label_dir):
                    # 尝试平行目录
                    label_dir = os.path.join(dataset_root, split.replace("images", ""), "labels")
                    if not os.path.exists(label_dir):
                        label_dir = os.path.join(dataset_root, "labels", split)

                for img_path in img_list:
                    # 读取图片尺寸以计算绝对像素面积
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    
                    # 寻找对应的 txt 标签
                    txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                    txt_path = os.path.join(label_dir, txt_name)
                    if not os.path.exists(txt_path):
                        # 尝试直接在同级目录
                        txt_path = os.path.join(os.path.dirname(img_path), txt_name)
                    
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    # YOLO 格式: class x_center y_center width height (归一化)
                                    bw_norm = float(parts[3])
                                    bh_norm = float(parts[4])
                                    
                                    abs_w = bw_norm * w
                                    abs_h = bh_norm * h
                                    area = abs_w * abs_h
                                    
                                    stats["total_instances"] += 1
                                    split_instances += 1
                                    
                                    # COCO 标准划分
                                    if area < 32 * 32:
                                        stats["small"] += 1
                                    elif area < 96 * 96:
                                        stats["medium"] += 1
                                    else:
                                        stats["large"] += 1
                
                stats["splits"][split]["instances"] = split_instances

    # 打印统计报告
    print(f"📌 【{dataset_name}】关键论文统计信息:")
    print(f"  - 总图片数 (Total Images)  : {stats['total_images']}")
    print(f"  - 总标注框 (Total Instances): {stats['total_instances']}")
    print(f"  - 各子集分布 (Splits)      : {stats['splits']}")
    
    total_inst = stats["total_instances"]
    if total_inst > 0:
        s_pct = (stats["small"] / total_inst) * 100
        m_pct = (stats["medium"] / total_inst) * 100
        l_pct = (stats["large"] / total_inst) * 100
        print(f"  - 目标尺度比例 (COCO Standard):")
        print(f"    * 小目标 (Small <32²px)   : {stats['small']} ({s_pct:.2f}%)")
        print(f"    * 中目标 (Medium 32²-96²px): {stats['medium']} ({m_pct:.2f}%)")
        print(f"    * 大目标 (Large >96²px)   : {stats['large']} ({l_pct:.2f}%)")
    else:
        print("  - ⚠️ 未检测到有效标注框或路径未匹配到标签文件。")
    print("=" * 60)

if __name__ == '__main__':
    datasets = {
     
        "Stage 4: micro_fire_only": r"F:\ultralyticsyolo8\datasets_stage3_small__D_fire"
    }

    for name, path in datasets.items():
        analyze_yolo_dataset(path, name)