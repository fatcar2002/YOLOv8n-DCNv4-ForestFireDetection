import os
import sys
from ultralytics import YOLO

# =====================================================================
# Stage 3: YOLOv8 + NWD + P2 继承权重小目标训练脚本
# =====================================================================
yaml_config = "yolov8-p2-dfire.yaml"  # 挂载你提供的这个四头 P2 配置文件
dataset_yaml = r"F:\ultralyticsyolo8\datasets_stage3_small__D_fire\stage3_small_dfire.yaml"

project_name = "runs/detect"
exp_name = "exp3_stage3_yolov8_nwd_p2_small_dfire"

if __name__ == '__main__':
    print("================ 🚀 开始运行 Stage 3: YOLOv8 + NWD + P2 小目标极速训练 ================")
    print(f"📄 模型结构配置: {yaml_config}")
    print(f"📄 数据集配置  : {dataset_yaml}")
    print(f"📁 实验输出目录: {os.path.join(project_name, exp_name)}\n")

    # 1. 加载 P2 模型结构并继承预训练权重
    model = YOLO(yaml_config, task="detect").load("yolov8n.pt")
    model.args = {"use_nwd": True}  # 动态挂载 NWD 损失函数

    # 2. 5090 D 专属极速训练
    results = model.train(
        data=dataset_yaml,
        epochs=150,
        imgsz=640,
        batch=32,                 # 🚀 吃满显存，加速迭代
        workers=8,                # 🚀 CPU 多线程预加载
        device=0,
        amp=True,                 # 🚀 开启混合精度加速
        cache=True,               # 🚀 内存缓存图像
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    print("\n🎉 YOLOv8 + NWD + P2 训练完成！")
    print(f"🏆 最佳权重已保存至: {os.path.join(project_name, exp_name, 'weights', 'best.pt')}")