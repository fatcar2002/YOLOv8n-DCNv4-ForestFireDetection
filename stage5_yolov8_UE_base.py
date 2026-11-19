import os
import torch
from ultralytics import YOLO

# 🚀 开启显卡 TF32 算子极速加速 (RTX 5090 D)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    # 1. 采用官方标准 YOLOv8n 结构与官方 COCO 预训练权重
    model_yaml = "yolov8n.yaml"
    pretrained_weights = "yolov8n.pt"  # 🎯 继承 100% 官方预训练权重
    
    # 🎯 指向刚才生成的新数据集配置文件路径
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_Ue_middlesmall_target\dataset.yaml"
    
    project_name = "runs/detect"
    exp_name = "stage5_yolov8_UE_base"  # 🎯 对应要求的命名

    print("================ 🚀 启动 YOLOv8 原版基线模型训练 (stage5_yolov8_UE_base) ================")

    # 加载预训练模型权重
    model = YOLO(model_yaml).load(pretrained_weights)

    # 2. 启动 100 代训练
    results = model.train(
        data=dataset_yaml,
        epochs=100,                 # 标准 100 代训练
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        amp=False,
        cache=True,
        optimizer="AdamW",          # 使用 AdamW 优化器
        lr0=0.001,                  # 标准初始学习率
        cos_lr=True,                # 开启余弦退火学习率
        close_mosaic=10,            # 最后 10 代关闭 Mosaic 数据增强
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    print("\n🎉 YOLOv8 原版基线训练完成！结果保存在 runs/detect/stage5_yolov8_UE_base")