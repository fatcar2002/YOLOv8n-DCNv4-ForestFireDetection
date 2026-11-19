import torch
from ultralytics import YOLO

# 🚀 开启显卡算子极速加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    # 1. 预训练权重路径与数据集配置
    pretrained_weights = r"F:\ultralyticsyolo8\weights\yolo26n.pt"  # 👈 确认你的 yolo26n.pt 路径
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_stage4_micro_fire_only\micro_fire_only.yaml" # 👈 指向刚建好的纯小火焰数据集

    project_name = "runs/detect"
    exp_name = "exp4_yolo26n_base_1epoch_test_small"

    print("================ 🚀 启动原版 YOLO26n 预训练权重 1 代 (1 Epoch) 测试 ================")

    # 2. 直接使用官方 API 加载预训练模型
    model = YOLO(pretrained_weights)

    # 3. 启动训练 1 代
    results = model.train(
        data=dataset_yaml,
        epochs=100,                  # 🎯 仅跑 1 代
        imgsz=640,
        batch=32,                  # 5090 D 可设为 32
        workers=8,                 # 防止 Windows 多进程死锁
        device=0,
        amp=False,                 # 保持 FP32/TF32
        cache=True,                # 开启缓存
        
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    print("\n🎉 原版 YOLO26n 1 代训练测试完成！")