import os
import torch
from ultralytics import YOLO

# 🚀 开启显卡 TF32 算子极速加速 (RTX 5090 D)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    # 🎯 直接加载权重文件（内部已包含网络结构，无需单独写 yaml）
    model_path = r"weights\yolo26n.pt"   # 或者直接写 "yolov26n.pt"
    
    # 🎯 指向数据集配置文件路径
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_D-fire-pure-small\pure_fire.yaml"
    
    project_name = "runs/detect"
    exp_name = "stage6_yolov26_2+1small_base"

    print("================ 🚀 启动 YOLO 基线模型训练 ================")

    # 直接通过权重实例化模型（完美避开 yaml 找不到的报错）
    model = YOLO(model_path)

    # 启动 100 代训练
    results = model.train(
        data=dataset_yaml,
        epochs=100,                 
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        amp=False,
        cache=True,
        optimizer="AdamW",          
        lr0=0.001,                  
        cos_lr=True,                
        close_mosaic=10,            
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )