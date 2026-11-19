import os
import pandas as pd
from ultralytics import YOLO

# 1. 独立配置权重路径（不覆盖之前的 runs/detect/ 评估日志）
weights_info = {
    "Stage 1 (OHID-FF 遥感数据集)": {
        "data_yaml": "ohid_ff.yaml",
        "models": {
            "Baseline (YOLOv8n)": "runs/detect/exp1_stage1_vanilla_baseline_ohid/weights/best.pt",
            "YOLO26n Benchmark": "runs/detect/exp1_stage1_yolo26_ohid/weights/best.pt",
            "Ours (P2+DCNv4+NWD)": "runs/detect/exp1_stage1_ours_ohid_DCNV4/weights/best.pt"
        }
    },
    "Stage 2 (D-Fire 近地/无人机数据集)": {
        "data_yaml": "dfire.yaml",
        "models": {
            "Baseline (YOLOv8n)": "runs/detect/exp2_stage2_baseline_dfire/weights/best.pt",
            "YOLO11n Baseline": "runs/detect/exp2_stage2_yolo11_dfire/weights/best.pt",
            "YOLO26n Benchmark": "runs/detect/exp2_stage2_yolo26_dfire/weights/best.pt",
            "Ours (P2+DCNv4+NWD)": "runs/detect/exp2_stage2_ours_dfire/weights/best.pt"
        }
    }
}

if __name__ == '__main__':
    print("================ 开始跑 YOLO11 / 通用 SOTA / Ours 小目标细分指标 (AP_small) ================")

    for stage_name, stage_data in weights_info.items():
        print(f"\n>>>>>> 当前测试数据集: {stage_name} (配置文件: {stage_data['data_yaml']}) <<<<<<")
        
        for exp_name, weight_path in stage_data["models"].items():
            if not os.path.exists(weight_path):
                print(f"⚠️ [跳过] 未找到权重文件: {weight_path} (请确认实验文件夹名称是否一致)")
                continue
                
            print(f"\n正在评估 [{exp_name}] -> 路径: {weight_path}")
            model = YOLO(weight_path)
            
            # 核心隔离与参数设置：
            # 1. project="runs/val_small_targets_yolo11": 结果独立保存在新文件夹，绝对不覆盖旧结果！
            # 2. save_json=True: 强制激活 pycocotools 打印 AP_small / AP_medium / AP_large
            # 3. workers=0: 规避 Windows 下 DataLoader 多线程崩溃问题
            metrics = model.val(
                data=stage_data['data_yaml'],
                split='test',
                save_json=True,
                batch=16,
                imgsz=640,
                workers=0,
                verbose=False,
                project="runs/val_small_targets_yolo11",
                name=exp_name.replace(" ", "_").replace("+", "_plus_"),
                exist_ok=True
            )
            print(f"✅ [{exp_name}] 评估完成！请留意控制台上方的 pycocotools 汇总表格中的 AP_small。")

    print("\n================ 全部模型小目标评估完成！日志已独立保存在 runs/val_small_targets_yolo11/ ================")