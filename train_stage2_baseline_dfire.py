from ultralytics import YOLO

if __name__ == "__main__":
    yaml_config = r"F:\ultralyticsyolo8\yolov8n-base-dfire.yaml"
    
    exp_name = "exp2_stage2_baseline_dfire"

    model = YOLO(yaml_config, task="detect").load("yolov8n.pt")
    model.args = {"use_nwd": False}

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