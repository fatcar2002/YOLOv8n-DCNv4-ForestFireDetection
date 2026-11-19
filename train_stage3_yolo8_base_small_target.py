import os
import sys
from ultralytics import YOLO

# =====================================================================
# 1. 配置 Stage 3 数据集与输出路径
# =====================================================================
dataset_yaml = r"F:\ultralyticsyolo8\datasets_stage4_micro_fire_only\micro_fire_only.yaml"
project_name = "runs/detect"
exp_name = "exp4_stage3_yolov8_base_small_dfire"

if __name__ == '__main__':
    print("================ 🚀 开始运行 Stage 3: YOLOv8 Baseline 小目标极速训练 ================")
    print(f"📄 数据集配置: {dataset_yaml}")
    print(f"📁 实验输出目录: {os.path.join(project_name, exp_name)}\n")

    # 2. 加载 YOLOv8 官方 Baseline 模型 (选用 yolov8n.pt 或 yolov8s.pt)
    model = YOLO("yolov8n.pt") 

    # 3. 开启 5090 D 专属极速训练
    results = model.train(
        data=dataset_yaml,
        epochs=100,               # 训练轮次
        imgsz=640,                # 输入图像尺寸
        batch=32,                 # 🚀 充分利用 5090 D 显存，加速迭代
        workers=8,                # 🚀 开启 CPU 多线程数据加载
        device=0,                 # 使用 GPU 0
        amp=True,                 # 🚀 开启混合精度训练 (Automatic Mixed Precision)
        cache=True,               # 🚀 将图像缓存到内存，读取速度翻倍
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    print("\n🎉 YOLOv8 Baseline 训练完成！")
    print(f"🏆 最佳权重已保存至: {os.path.join(project_name, exp_name, 'weights', 'best.pt')}")