import os
import glob
import pandas as pd

def analyze_full_dataset(dataset_name, dataset_root):
    print(f"\n================ 📊 正在深度分析数据集: {dataset_name} ================")
    if not os.path.exists(dataset_root):
        print(f"⚠️ 路径不存在: {dataset_root}")
        return None

    splits = ['train', 'val', 'test']
    summary_rows = []

    grand_total_images = 0
    grand_total_instances = 0
    grand_small = 0
    grand_medium = 0
    grand_large = 0

    for split in splits:
        # 兼容不同的 YOLO 目录结构
        img_dir_candidates = [
            os.path.join(dataset_root, 'images', split),
            os.path.join(dataset_root, split, 'images'),
            os.path.join(dataset_root, split)
        ]
        
        img_dir = None
        for cand in img_dir_candidates:
            if os.path.exists(cand) and os.path.isdir(cand):
                img_dir = cand
                break
                
        if not img_dir:
            continue
            
        # 定位标签目录
        lbl_dir = None
        for cand_lbl in [
            img_dir.replace('images', 'labels'),
            os.path.join(dataset_root, 'labels', split),
            os.path.join(dataset_root, split, 'labels')
        ]:
            if cand_lbl and os.path.exists(cand_lbl):
                lbl_dir = cand_lbl
                break

        images = glob.glob(os.path.join(img_dir, '*.jpg')) + \
                 glob.glob(os.path.join(img_dir, '*.png')) + \
                 glob.glob(os.path.join(img_dir, '*.jpeg'))
        
        num_images = len(images)
        num_instances = 0
        small, medium, large = 0, 0, 0
        
        for img_path in images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            if not lbl_dir:
                continue
            lbl_path = os.path.join(lbl_dir, base_name + '.txt')
            
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    num_instances += len(lines)
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                w_norm = float(parts[3])
                                h_norm = float(parts[4])
                                # 假设 640x640 基准计算像素面积
                                pixel_area = (w_norm * 640) * (h_norm * 640)
                                
                                if pixel_area < 32 * 32:
                                    small += 1
                                elif pixel_area < 96 * 96:
                                    medium += 1
                                else:
                                    large += 1
                            except ValueError:
                                continue

        grand_total_images += num_images
        grand_total_instances += num_instances
        grand_small += small
        grand_medium += medium
        grand_large += large

        summary_rows.append({
            "Split": split.upper(),
            "Images": num_images,
            "Instances": num_instances,
            "Small (<32²)": f"{small} ({small/max(1,num_instances)*100:.1f}%)",
            "Medium (32²-96²)": f"{medium} ({medium/max(1,num_instances)*100:.1f}%)",
            "Large (>96²)": f"{large} ({large/max(1,num_instances)*100:.1f}%)"
        })

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))
    print(f"  📌 【{dataset_name} 总体】总图像: {grand_total_images} | 总实例: {grand_total_instances} | 小目标占比: {grand_small/max(1,grand_total_instances)*100:.2f}%")
    return grand_total_images, grand_total_instances

if __name__ == '__main__':
    # 你的核心数据集路径
    datasets_to_analyze = {
        "datasets_stage3_small__D_fire": r"F:\ultralyticsyolo8\datasets_stage3_small__D_fire",
        "D-Fire-Pure-Small": r"F:\ultralyticsyolo8\datasets_D-fire-pure-small",
        "Stage2-D-Fire": r"F:\ultralyticsyolo8\datasets_stage2_D-fire",
        "Stage4-Micro-Fire": r"F:\ultralyticsyolo8\datasets_stage4_micro_fire_only",
        "UE-Middle-Small": r"F:\ultralyticsyolo8\datasets_Ue_middlesmall_target",
        "ohid":r"F:\ultralyticsyolo8\dataset_stage1_ohid"
    }

    for name, path in datasets_to_analyze.items():
        analyze_full_dataset(name, path)

    print("\n================ 🏆 所有数据集（含 Test 集）统计完成 ================")