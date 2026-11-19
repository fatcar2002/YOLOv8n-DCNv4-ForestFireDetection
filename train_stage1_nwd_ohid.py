from ultralytics import YOLO

if __name__ == "__main__":
    # 使用与 Baseline 相同的网络拓扑，单独开启 NWD Loss
    yaml_config = "yolov8n-NWD-ohid.yaml"
    exp_name = "exp1_stage1_nwd_ohid"

    model = YOLO(yaml_config, task="detect")
    
    # 显式开启 NWD Loss
    model.args = {"use_nwd": True}

    results = model.train(
        data="ohid_ff.yaml",
        epochs=150,
        imgsz=640,
        batch=64,
        lr0=0.0005,
        warmup_epochs=10.0,
        optimizer="SGD",
        amp=False,
        device=0,
        name=exp_name,
        exist_ok=True
    )