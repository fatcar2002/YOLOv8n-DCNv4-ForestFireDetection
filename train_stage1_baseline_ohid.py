from ultralytics import YOLO

if __name__ == "__main__":
    # 使用纯 Baseline YAML 配置
    yaml_config = "yolov8n-base-ohid.yaml"
    exp_name = "exp1_stage1_baseline_ohid"

    model = YOLO(yaml_config, task="detect")
    
    # 强制关闭 NWD Loss，使用原生 CIoU Loss
    model.args = {"use_nwd": False}

    results = model.train(
        data="ohid_ff.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        lr0=0.0005,
        warmup_epochs=10.0,
        optimizer="SGD",
        amp=False,
        device=0,
        name=exp_name,
        exist_ok=True
    )