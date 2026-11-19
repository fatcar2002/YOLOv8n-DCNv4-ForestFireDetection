from ultralytics import YOLO

if __name__ == "__main__":
    model_name = r"weights\yolo26n.pt"
    exp_name = "exp1_stage1_yolo26_ohid"

    model = YOLO(model_name)

    results = model.train(
        data="ohid_ff.yaml",        # 遥感数据集
        epochs=150,                 # 150 轮
        imgsz=640,
        batch=8,                   # 建议从 8 调大到 16（5090 D 显存充裕，进一步提升 GPU 利用率）
        lr0=0.0005,
        warmup_epochs=10.0,
        optimizer="SGD",
        amp=False,                  # 保持公平对比
        device=0,
        name=exp_name,
        exist_ok=True,
        workers=8                   # 【关键优化】开启 8 个 CPU 子线程并行加载数据！
    )