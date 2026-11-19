import os
import glob
import copy
import sys
import pandas as pd
from ultralytics import YOLO

# =====================================================================
# 1. 挂载真实 DCNv4 算子（保证验证正常）
# =====================================================================
try:
    import dcnv4_module
    from dcnv4_module import DCNv4_2D
    sys.modules["DCNv4"] = dcnv4_module
    sys.modules["DCNv4.modules"] = dcnv4_module
    sys.modules["DCNv4.modules.dcnv4"] = dcnv4_module
    sys.modules["dcnv4_module"] = dcnv4_module
except Exception:
    pass

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules

_orig_parse_model = tasks.parse_model
def patched_parse_model(d, ch, verbose=True):
    d = copy.deepcopy(d)
    dcn_layers = {}
    layers = d.get('backbone', []) + d.get('head', [])
    for i, layer in enumerate(layers):
        if layer[2] == 'DCNv4_2D':
            dcn_layers[i] = layer[3]
            layer[2] = 'Conv'
    model_seq, save_list = _orig_parse_model(d, ch, verbose)
    for i, dcn_args in dcn_layers.items():
        placeholder_conv = model_seq[i]
        c1, c2 = placeholder_conv.conv.in_channels, placeholder_conv.conv.out_channels
        k = dcn_args[1] if len(dcn_args) > 1 else 3
        s = dcn_args[2] if len(dcn_args) > 2 else 1
        new_m = DCNv4_2D(c1, c2, k=k, s=s)
        new_m.f, new_m.i, new_m.type = getattr(placeholder_conv, 'f', -1), getattr(placeholder_conv, 'i', i), 'DCNv4_2D'
        model_seq[i] = new_m
    return model_seq, save_list

tasks.parse_model = patched_parse_model
setattr(modules, "DCNv4_2D", DCNv4_2D)
setattr(modules.block, "DCNv4_2D", DCNv4_2D)
setattr(tasks, "DCNv4_2D", DCNv4_2D)

# =====================================================================
# 2. 统计数据集中标注框的真实尺寸分布 (按 COCO 640x640 面积标准)
# =====================================================================
def analyze_dataset_objects(labels_dir, imgsz=640):
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    total_boxes = 0
    small_cnt = 0   # 面积 < 32^2 (1024)
    medium_cnt = 0  # 32^2 <= 面积 <= 96^2 (9216)
    large_cnt = 0   # 面积 > 96^2

    for file in txt_files:
        with open(file, 'r') as f:
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

    return total_boxes, small_cnt, medium_cnt, large_cnt

# =====================================================================
# 3. 运行 12 组实验的中小目标评估提取
# =====================================================================
weights_info = {
    "Stage 1 (OHID-FF 遥感)": {
        "data_yaml": "ohid_ff.yaml",
        "labels_dir": r"F:\ultralyticsyolo8\dataset_ohid\labels\test", # 请替换为实际的 test 标注文件夹路径
        "models": [
            {"name": "v8 Baseline (100%权重)", "path": "runs/detect/exp1_stage1_vanilla_baseline_ohid/weights/best.pt"},
            {"name": "v8 +NWD (有权重)", "path": "runs/detect/exp1_stage1_p2_ohid/weights/best.pt"},
            {"name": "v8 +P2+NWD (有权重)", "path": "runs/detect/exp1_stage1_ohid_ours/weights/best.pt"},
            {"name": "v8 Ours (完全体有权重)", "path": "runs/detect/exp1_stage1_ours_ohid_DCNV4/weights/best.pt"},
            {"name": "v26 Baseline (0权重 Scratch)", "path": "runs/detect/exp1_stage1_yolo26_scratch_ohid/weights/best.pt"},
            {"name": "v26 Baseline (有权重)", "path": "runs/detect/exp1_stage1_yolo26_ohid/weights/best.pt"},
            {"name": "v26 Ours (0权重 Scratch)", "path": "runs/detect/exp1_stage1_yolo26_ours_scratch_ohid/weights/best.pt"},
        ]
    },
    "Stage 2 (D-Fire 近地/无人机)": {
        "data_yaml": "dfire.yaml",
        "labels_dir": r"F:\ultralyticsyolo8\dataset_dfire\labels\test", # 请替换为实际的 test 标注文件夹路径
        "models": [
            {"name": "v8 Baseline (100%权重)", "path": "runs/detect/exp2_stage2_baseline_dfire/weights/best.pt"},
            {"name": "v8 +NWD (有权重)", "path": "runs/detect/exp2_stage2_nwd_dfire/weights/best.pt"},
            {"name": "v8 +P2+NWD (有权重)", "path": "runs/detect/exp2_stage2_p2_nwd_dfire/weights/best.pt"},
            {"name": "v8 Ours (标准完全体有权重)", "path": "runs/detect/exp2_stage2_ours_dfire/weights/best.pt"},
            {"name": "v8 Ours (智能切片57%最佳有权重)", "path": "runs/detect/exp2_stage2_ours_smart_aligned_test/weights/best.pt"},
            {"name": "v26 Baseline (0权重 Scratch)", "path": "runs/detect/exp2_stage2_yolo26_dfire_scratch/weights/best.pt"},
            {"name": "v26 Baseline (有权重)", "path": "runs/detect/exp2_stage2_yolo26_dfire/weights/best.pt"},
            {"name": "v26 Ours (0权重 Scratch)", "path": "runs/detect/exp2_stage2_yolo26_ours_scratch_dfire/weights/best.pt"},
        ]
    }
}

if __name__ == '__main__':
    print("================ 📊 开始统计数据集目标分布与中小目标检测指标 ================\n")

    for stage_name, stage_data in weights_info.items():
        lbl_dir = stage_data["labels_dir"]
        if os.path.exists(lbl_dir):
            tot, s_cnt, m_cnt, l_cnt = analyze_dataset_objects(lbl_dir)
            s_ratio = (s_cnt / tot * 100) if tot > 0 else 0
            m_ratio = (m_cnt / tot * 100) if tot > 0 else 0
            sm_ratio = ((s_cnt + m_cnt) / tot * 100) if tot > 0 else 0
            
            print(f"📌 [{stage_name}] 测试集目标分布统计:")
            print(f"   ├─ 标注框总数: {tot}")
            print(f"   ├─ 极小目标 (Small, <32²): {s_cnt} 个 | 占比: {s_ratio:.2f}%")
            print(f"   ├─ 中等目标 (Medium, 32²~96²): {m_cnt} 个 | 占比: {m_ratio:.2f}%")
            print(f"   └─ 中小目标合计 (Small+Medium): {s_cnt + m_cnt} 个 | 总体占比: {sm_ratio:.2f}%\n")
        else:
            print(f"⚠️ 未找到标签路径 {lbl_dir}，请在脚本中检查实际的 labels/test 目录路径。\n")

    print("================ 🎯 正在提取各模型 AP_small (小目标) 与 AP_medium (中目标) 指标 ================")

    small_results = []
    for stage_name, stage_data in weights_info.items():
        for item in stage_data["models"]:
            exp_name, weight_path = item["name"], item["path"]
            if not os.path.exists(weight_path):
                continue
                
            model = YOLO(weight_path)
            metrics = model.val(
                data=stage_data["data_yaml"],
                split='test',
                save_json=True,
                batch=8,
                imgsz=640,
                workers=0,
                verbose=False
            )
            
            coco_stats = getattr(metrics, 'coco_stat', None)
            if coco_stats is not None and len(coco_stats) >= 5:
                aps = f"{coco_stats[3]*100:.2f}"   # AP_small
                apm = f"{coco_stats[4]*100:.2f}"   # AP_medium
            else:
                aps, apm = "查看控制台", "查看控制台"

            small_results.append({
                "数据集场景": stage_name,
                "模型变体": exp_name,
                "AP_small (小目标 %)": aps,
                "AP_medium (中目标 %)": apm
            })

    print("\n================ 📋 论文专用：中小目标检测性能汇总表 ================")
    df = pd.DataFrame(small_results)
    print(df.to_string(index=False))