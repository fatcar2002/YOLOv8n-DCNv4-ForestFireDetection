from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 0权重从头训练（Scratch）：直接加载模型的架构 yaml 文件，不载入任何 .pt 权重
    # （请根据你本地实际的 yolo26 配置文件名调整，例如 "yolo26n.yaml"）
    model_name = r"yolo26n.yaml"  
    exp_name = "exp2_stage2_yolo26_dfire_scratch"

    model = YOLO(model_name)

    # 2. 启动训练，所有超参数严格与之前保持 100% 对齐
    results = model.train(
        data="dfire.yaml",          # D-Fire 数据集配置文件
        epochs=150,                 # 保持 150 轮[cite: 6]
        imgsz=640,                  # 640x640 输入尺寸[cite: 6]
        batch=8,                    # 保持 batch=8 严格控制变量[cite: 6]
        lr0=0.0005,                 # 学习率[cite: 6]
        warmup_epochs=10.0,         # 10 轮预热[cite: 6]
        optimizer="SGD",            # SGD 优化器[cite: 6]
        amp=False,                  # 关闭混合精度[cite: 6]
        device=0,                   # 指定显卡 0[cite: 6]
        name=exp_name,              # 输出结果文件夹名
        exist_ok=True,              # 允许同名文件夹覆盖[cite: 6]
        workers=8                   # 开启 8 个 CPU 子线程
    )