from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 0 权重从头训练（Scratch）：传入 YAML 架构配置，不载入任何 .pt 权重文件
    # （请确认本地对应的配置文件名，若为单类烟雾，请确保 yolo26n.yaml 中 nc: 1）
    model_name = "yolo26n.yaml"
    exp_name = "exp1_stage1_yolo26_scratch_ohid"

    model = YOLO(model_name)

    # 2. 启动遥感数据集（OHID-FF）150 轮训练
    results = model.train(
        data="ohid_ff.yaml",        # 遥感数据集配置文件[cite: 13]
        epochs=150,                 # 150 轮[cite: 13]
        imgsz=640,                  # 640x640[cite: 13]
        batch=8,                    # 保持 batch=8 严格控制变量[cite: 13]
        lr0=0.0005,                 # 学习率 0.0005[cite: 13]
        warmup_epochs=10.0,         # 10 轮预热[cite: 13]
        optimizer="SGD",            # SGD 优化器[cite: 13]
        amp=False,                  # 关闭混合精度[cite: 13]
        device=0,                   # 指定显卡 0[cite: 13]
        name=exp_name,              # 实验命名
        exist_ok=True,              # 覆盖同名文件夹[cite: 13]
        workers=8                   # 8 个子线程[cite: 13]
    )