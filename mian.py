import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载当前目录下的预训练权重
    model = YOLO('yolov8n.pt')

    # 2. 启动基础训练测试（适配 5090 D 硬件）
    results = model.train(
        data='coco8.yaml',   # 使用本地 coco8 数据集
        epochs=3,            # 快速跑 3 轮验证流程
        imgsz=640,
        batch=16,
        device=0,            # 明确指定使用 RTX 5090 D
        amp=False,           # 禁用联网 AMP 检查，防止连接超时
        project='runs/train',
        name='yolov8_baseline_test'
    )
    
    print("--------------------------------------------------")
    print("YOLOv8 在 RTX 5090 D 上的 Baseline 训练成功跑通！")
    print("--------------------------------------------------")