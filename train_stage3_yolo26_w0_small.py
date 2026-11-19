import torch
from ultralytics import YOLO

# 🚀 开启 TF32 模式：在 FP16/FP32 下全面激活 RTX 5090 D 的 Tensor Core 算力
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    # 1. 0 权重从头训练（Scratch）：只加载 YOLO26 的架构 yaml 配置文件，不载入任何 .pt 权重
    yaml_config = "yolo26n.yaml"  # 请确保你本地有 yolo26n.yaml 配置文件
    
    # 2. 指向 Stage 3 小目标数据集
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_stage3_small__D_fire\stage3_small_dfire.yaml"

    project_name = "runs/detect"
    exp_name = "exp0_stage3_yolo26n_baseline_scratch_amp16"

    print("================ 🚀 开始运行基线对比组: YOLO26n Original (0权重/Scratch, FP16) ================")
    
    # 初始化原版 YOLO26 模型结构
    model = YOLO(yaml_config, task="detect")

    # 3. 启动训练（严格控制变量）
    results = model.train(
        data=dataset_yaml,
        epochs=150,
        imgsz=640,
        batch=16,                 # 🚀 保持 batch=16 严格控制变量
        workers=0,                # 🚀 必须设为 0，彻底防止 Windows 多进程假死
        device=0,
        amp=True,                 # 🚀 开启 FP16 混合精度（原版 YOLO26 算子可安全开启）
        cache=False,              # 🚀 保持 False，防止物理内存溢出
        lr0=0.0005,               # 保持 YOLO26 官方推荐的 0.0005 初始学习率
        warmup_epochs=10.0,
        optimizer="SGD",
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )