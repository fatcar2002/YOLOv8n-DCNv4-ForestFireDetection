import torch
from ultralytics import YOLO

# 🚀 开启 TF32 极速模式（RTX 5090 D 专属加速）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    # 🎯 Base 组核心：直接加载官方标准 yolov8n.pt 模型（原生结构 + 原生预训练权重）
    model_path = r"F:\ultralyticsyolo8\yolov8n.pt"
    
    # 🎯 对应你构建的 300 张 Mini 小目标精简数据集 YAML
    dataset_yaml = r"F:\ultralyticsyolo8\datasets300test_small\stage3_mini300_dfire.yaml"

    project_name = "runs/detect"
    exp_name = "exp_mini300_base_yolov8n"

    print("================ 🚀 开始运行: Standard Base YOLOv8n (300张 Mini 数据集 80轮) ================")
    
    # 1. 直接实例化官方标准的 Base YOLOv8 模型
    model = YOLO(model_path, task="detect")

    # 2. 启动训练（与 Ours 组保持严格相同的超参数以保证消融实验公平性）
    results = model.train(
        data=dataset_yaml,
        epochs=80,                 # 🎯 同样跑 80 轮，与 Ours 组形成严格变量控制
        imgsz=640,
        batch=16,                  # 显存安全 Batch
        workers=0,                 # 设为 0，防止 Windows 多进程死锁
        device=0,
        amp=False,                 # 保持 FP32/TF32 全精度模式，保证计算精度一致
        cache=False,
        lr0=0.0005,                # 保持与 Ours 组相同的初始学习率
        lrf=0.01,
        warmup_epochs=3.0,         # 同样 3 轮预热
        cos_lr=True,               # 显式开启余弦退火
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )