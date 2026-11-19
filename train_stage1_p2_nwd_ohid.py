from ultralytics import YOLO

if __name__ == "__main__":
    # 使用 P2 浅层四头拓扑结构
    yaml_config = "yolov8-p2-ohid.yaml"
    exp_name = "exp1_stage1_p2_nwd_ohid"

    model = YOLO(yaml_config, task="detect")
    
    # 显式开启 NWD Loss
    model.args = {"use_nwd": True}

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