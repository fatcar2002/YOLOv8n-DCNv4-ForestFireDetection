import os
import torch
from ultralytics import YOLO

# 必须加上入口保护，规避 Windows spawn 机制
if __name__ == '__main__':
    # 强制开启 GPU 0 (CUDA)
    device_id = 0 if torch.cuda.is_available() else 'cpu'
    
    night_yaml = r"F:\ultralyticsyolo8\dataset_night_test\dataset_night_test.yaml"
    baseline_weight = "runs/detect/exp2_stage2_baseline_dfire/weights/best.pt"
    ours_weight = "runs/detect/exp2_stage2_ours_dfire/weights/best.pt"

    print(f"================ 开始评估夜间场景 (GPU 设备: {device_id}) ================")

    # 1. 评估 Baseline
    if os.path.exists(baseline_weight):
        print(f"\n>>>> 正在评估 Baseline (YOLOv8n) <<<<")
        model_base = YOLO(baseline_weight)
        m_base = model_base.val(
            data=night_yaml,
            split='test',
            batch=16,
            imgsz=640,
            device=device_id,
            workers=0,
            verbose=False
        )
        print(f"📊 Baseline 结果 -> Precision: {m_base.box.mp*100:.2f}%, Recall: {m_base.box.mr*100:.2f}%, mAP50: {m_base.box.map50*100:.2f}%, mAP50-95: {m_base.box.map*100:.2f}%")

    # 2. 评估 Ours (完全体)
    if os.path.exists(ours_weight):
        print(f"\n>>>> 正在评估 Ours (P2 + DCNv4 + NWD) <<<<")
        model_ours = YOLO(ours_weight)
        m_ours = model_ours.val(
            data=night_yaml,
            split='test',
            batch=16,
            imgsz=640,
            device=device_id,
            workers=0,
            verbose=False
        )
        print(f"📊 Ours 结果     -> Precision: {m_ours.box.mp*100:.2f}%, Recall: {m_ours.box.mr*100:.2f}%, mAP50: {m_ours.box.map50*100:.2f}%, mAP50-95: {m_ours.box.map*100:.2f}%")

    print("\n================ 夜间场景评估完毕 ================")