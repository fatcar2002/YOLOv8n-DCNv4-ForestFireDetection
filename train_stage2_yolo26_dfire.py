from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 加载预训练权重 (与 Stage 1 保持一致，从本地 weights 目录载入)
    model_name = r"weights\yolo26n.pt"
    exp_name = "exp2_stage2_yolo26_dfire"

    model = YOLO(model_name)

    # 2. 启动训练，所有超参数严格与 Stage 2 D-Fire Baseline 保持 100% 对齐
    results = model.train(
        data="dfire.yaml",          # D-Fire 数据集配置文件
        epochs=150,                 # 保持 150 轮[cite: 6]
        imgsz=640,                  # 640x640 输入尺寸[cite: 6]
        batch=8,                    # 保持 batch=8 严格控制变量[cite: 6]
        lr0=0.0005,                 # 与 Baseline 保持一致的学习率[cite: 6]
        warmup_epochs=10.0,         # 10 轮预热[cite: 6]
        optimizer="SGD",            # SGD 优化器[cite: 6]
        amp=False,                  # 关闭混合精度以保障评估公平性[cite: 6]
        device=0,                   # 指定显卡 0[cite: 6]
        name=exp_name,              # 输出结果文件夹名 exp2_stage2_yolo26_dfire
        exist_ok=True,              # 覆盖/允许同名文件夹[cite: 6]
        workers=8                   # 开启 8 个 CPU 子线程，彻底解除 DataLoader 读取瓶颈
    )   