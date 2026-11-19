import os
import glob
import pandas as pd

# 配置你指定的标签文件夹路径
labels_dir = r"F:\人工智能相关备份\训练集2\黑烟标记2500"
imgsz = 640  # COCO 评估标准基准分辨率

def count_dataset_objects(lbl_dir, imgsz=640):
    txt_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    
    total_files = len(txt_files)
    total_boxes = 0
    small_cnt = 0   # 面积 < 32^2 (1024 像素²)
    medium_cnt = 0  # 32^2 <= 面积 <= 96^2 (9216 像素²)
    large_cnt = 0   # 面积 > 96^2 (9216 像素²)

    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w = float(parts[3]) * imgsz
                    h = float(parts[4]) * imgsz
                    area = w * h
                    total_boxes += 1
                    
                    if area < 1024:
                        small_cnt += 1
                    elif area <= 9216:
                        medium_cnt += 1
                    else:
                        large_cnt += 1

    return total_files, total_boxes, small_cnt, medium_cnt, large_cnt

if __name__ == '__main__':
    print(f"================ 🔍 正在深度统计数据集目标尺寸分布 ================")
    print(f"📁 统计目标目录: {labels_dir}\n")

    if not os.path.exists(labels_dir):
        print(f"⚠️ 找不到路径：{labels_dir}，请检查盘符或文件夹名称是否正确！")
    else:
        num_files, tot_boxes, s_cnt, m_cnt, l_cnt = count_dataset_objects(labels_dir, imgsz)

        if tot_boxes == 0:
            print("⚠️ 目录下未扫描到有效的标注框！")
        else:
            s_ratio = (s_cnt / tot_boxes) * 100
            m_ratio = (m_cnt / tot_boxes) * 100
            l_ratio = (l_cnt / tot_boxes) * 100
            sm_ratio = ((s_cnt + m_cnt) / tot_boxes) * 100

            print(f"📊 统计结果报告 (基于 COCO 640x640 面积划分标准):")
            print(f"--------------------------------------------------")
            print(f"  • 标注文件总数 (TXT 数量) : {num_files} 个")
            print(f"  • 标注框总实例数 (Boxes)   : {tot_boxes} 个")
            print(f"--------------------------------------------------")
            print(f"  🔹 极小/小目标 (Small, < 32²)        : {s_cnt} 个  | 占比: {s_ratio:.2f}%")
            print(f"  🔸 中等目标    (Medium, 32² ~ 96²)   : {m_cnt} 个  | 占比: {m_ratio:.2f}%")
            print(f"  🔴 大目标      (Large, > 96²)        : {l_cnt} 个  | 占比: {l_ratio:.2f}%")
            print(f"--------------------------------------------------")
            print(f"  💡 中小目标合计 (Small + Medium)     : {s_cnt + m_cnt} 个 | 总体占比: {sm_ratio:.2f}%\n")