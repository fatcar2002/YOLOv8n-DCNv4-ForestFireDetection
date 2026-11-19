import os
import pandas as pd
from ultralytics import YOLO

# 根据你 runs\detect 下的真实文件夹名称配置精确路径
weights_info = {
    "Stage 1 (OHID-FF 遥感)": {
        "data_yaml": "ohid_ff.yaml",
        "models": {
            "Baseline": "runs/detect/exp1_stage1_vanilla_baseline_ohid/weights/best.pt",
            "+NWD": "runs/detect/exp1_stage1_p2_ohid/weights/best.pt",
            "+P2+NWD": "runs/detect/exp1_stage1_ohid_ours/weights/best.pt",
            "Ours (P2+DCNv4+NWD)": "runs/detect/exp1_stage1_ours_ohid_DCNV4/weights/best.pt"
        }
    },
    "Stage 2 (D-Fire 近地/无人机)": {
        "data_yaml": "dfire.yaml",
        "models": {
            "Baseline": "runs/detect/exp2_stage2_baseline_dfire/weights/best.pt",
            "+NWD": "runs/detect/exp2_stage2_nwd_dfire/weights/best.pt",
            "+P2+NWD": "runs/detect/exp2_stage2_p2_nwd_dfire/weights/best.pt",
            "Ours (P2+DCNv4+NWD)": "runs/detect/exp2_stage2_ours_dfire/weights/best.pt"
        }
    }
}

# 必须加上入口保护，解决 Windows 下 spawn 机制报错
if __name__ == '__main__':
    print("================ 开始跑 8 个权重的 COCO 细分指标测试 ================")

    for stage_name, stage_data in weights_info.items():
        print(f"\n>>>>>> 当前测试数据集: {stage_name} (配置文件: {stage_data['data_yaml']}) <<<<<<")
        
        for exp_name, weight_path in stage_data["models"].items():
            if not os.path.exists(weight_path):
                print(f"⚠️ [跳过] 未找到权重文件: {weight_path}")
                continue
                
            print(f"\n正在评估 [{exp_name}] -> 路径: {weight_path}")
            model = YOLO(weight_path)
            
            # save_json=True 激活 pycocotools 评估，输出 AP_small / AP_medium / AP_large
            # workers=0 彻底规避 DataLoader 多进程潜在冲突
            metrics = model.val(
                data=stage_data["data_yaml"],
                split='test',
                save_json=True,
                batch=16,
                imgsz=640,
                workers=0,
                verbose=False
            )
            print(f"✅ [{exp_name}] 评估完成！请留意控制台上方的 pycocotools 汇总表格（含 AP_small）。")

    print("\n================ 全部 8 个实验评估完成 ================")