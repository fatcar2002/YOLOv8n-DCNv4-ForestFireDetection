from ultralytics import YOLO

if __name__ == "__main__":
    yaml_config = "yolov8-p2-dfire.yaml"
    exp_name = "exp2_stage2_p2_nwd_dfire"

    model = YOLO(yaml_config, task="detect").load("yolov8n.pt")
    model.args = {"use_nwd": True}

    results = model.train(
        data="dfire.yaml",
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