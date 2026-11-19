import os
import glob
from collections import defaultdict
import numpy as np

# 数据集根目录
dataset_root = r"F:\人工智能相关备份\训练集2\黑烟标记2500"
categories = {0: "fire", 1: "smoke"}  # 类别映射

# 定义 COCO 标准相对面积阈值 (基于 640x640 归一化坐标)
SMALL_THRESH = (32 / 640) ** 2   # 0.0025
LARGE_THRESH = (96 / 640) ** 2   # 0.0225

def analyze_split(split_name):
    label_dir = os.path.join(dataset_root, split_name, "labels")
    if not os.path.exists(label_dir):
        # 兼容图像与标签同目录或 labels 子目录的情况
        label_dir = os.path.join(dataset_root, split_name)
    
    txt_files = glob.glob(os.path.join(label_dir, "**", "*.txt"), recursive=True)
    
    stats = {
        "total_instances": 0,
        "small": 0,
        "medium": 0,
        "large": 0,
        "by_class": defaultdict(lambda: {"small": 0, "medium": 0, "large": 0, "total": 0, "areas": []})
    }
    
    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                area = w * h  # 相对面积 (0 ~ 1)
                
                stats["total_instances"] += 1
                cls_name = categories.get(cls_id, f"class_{cls_id}")
                stats["by_class"][cls_name]["total"] += 1
                stats["by_class"][cls_name]["areas"].append(area)
                
                if area < SMALL_THRESH:
                    stats["small"] += 1
                    stats["by_class"][cls_name]["small"] += 1
                elif area < LARGE_THRESH:
                    stats["medium"] += 1
                    stats["by_class"][cls_name]["medium"] += 1
                else:
                    stats["large"] += 1
                    stats["by_class"][cls_name]["large"] += 1
                    
    return stats

if __name__ == '__main__':
    print(f"================ 🔍 数据集尺寸分布分析: {dataset_root} ================\n")
    
    total_all = {"small": 0, "medium": 0, "large": 0, "total": 0}
    class_all = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0, "total": 0})
    
    for split in ["train", "val", "test"]:
        if os.path.exists(os.path.join(dataset_root, split)):
            res = analyze_split(split)
            print(f"📁 [{split.upper()} 集] 实例总数: {res['total_instances']}")
            print(f"   - 小目标 (<32x32): {res['small']} ({res['small']/max(1, res['total_instances'])*100:.2f}%)")
            print(f"   - 中目标 (32-96) : {res['medium']} ({res['medium']/max(1, res['total_instances'])*100:.2f}%)")
            print(f"   - 大目标 (>96x96): {res['large']} ({res['large']/max(1, res['total_instances'])*100:.2f}%)")
            
            total_all["total"] += res["total_instances"]
            total_all["small"] += res["small"]
            total_all["medium"] += res["medium"]
            total_all["large"] += res["large"]
            
            for cls_name, cls_data in res["by_class"].items():
                for k in ["small", "medium", "large", "total"]:
                    class_all[cls_name][k] += cls_data[k]
            print("-" * 60)
            
    print(f"\n🏆 【全数据集汇总结果】 (总目标数: {total_all['total']})")
    print(f"  🔹 小目标 (Small)  : {total_all['small']:6d} 个 ({total_all['small']/max(1, total_all['total'])*100:.2f}%)")
    print(f"  🔸 中目标 (Medium) : {total_all['medium']:6d} 个 ({total_all['medium']/max(1, total_all['total'])*100:.2f}%)")
    print(f"  🔻 大目标 (Large)  : {total_all['large']:6d} 个 ({total_all['large']/max(1, total_all['total'])*100:.2f}%)")
    
    print("\n🔥 【火焰 (Fire) vs 烟雾 (Smoke) 目标尺寸差异分析】")
    for cls_name, cls_data in class_all.items():
        tot = max(1, cls_data["total"])
        print(f"\n📌 类别 [{cls_name}] (总数: {cls_data['total']}):")
        print(f"   - 小目标比例: {cls_data['small']/tot*100:.2f}% ({cls_data['small']} 个)")
        print(f"   - 中目标比例: {cls_data['medium']/tot*100:.2f}% ({cls_data['medium']} 个)")
        print(f"   - 大目标比例: {cls_data['large']/tot*100:.2f}% ({cls_data['large']} 个)")