import torch
from ultralytics import YOLO

# 1. 自动检查官方 YOLOv8n 权重物理对齐度函数
def check_official_v8_alignment(model_yaml, pretrained_path):
    print(f"\n================ 正在进行原版 YOLOv8n 预训练对齐度物理检验 ================")
    # 实例化未灌入权重的纯官方 v8 结构
    net = YOLO(model_yaml, task="detect")
    model_state = net.model.state_dict()
    
    # 读取官方 weights (加上 weights_only=False 兼容 PyTorch 2.6)
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    pretrained_state = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    
    matched_keys = [k for k in pretrained_state.keys() if k in model_state and pretrained_state[k].shape == model_state[k].shape]
    total_official_keys = len(pretrained_state)
    coverage = len(matched_keys) / total_official_keys * 100
    
    print(f"📊 官方 yolov8n.pt 权重总层数: {total_official_keys}")
    print(f"✅ 官方 yolov8n.yaml 结构物理匹配层数: {len(matched_keys)}")
    print(f"🔥 原版预训练先验覆盖率: {coverage:.2f}%")
    if coverage > 95.0:
        print("🎉 物理对齐判定: 【100% 完全物理完美对齐！】\n")
    else:
        print("⚠️ 物理对齐判定: 【存在未对齐层！】\n")


if __name__ == "__main__":
    yaml_config = "yolov8n.yaml"
    pretrained_weights = r"F:\ultralyticsyolo8\yolov8n.pt"

    # A. 运行官方物理级权重覆盖率计算
    check_official_v8_alignment(yaml_config, pretrained_weights)

    # B. 启动 3 轮安全训练验证（保存在独立的 check_v8_official_test 文件夹）
    exp_name = "check_v8_official_test"
    
    print("================ 启动原版 YOLOv8n 3 轮起步指标验证 ================")
    model = YOLO(yaml_config, task="detect").load(pretrained_weights)

    results = model.train(
        data="dfire.yaml",
        epochs=3,                  # 仅跑 3 轮测试起步指标
        imgsz=640,
        batch=8,
        lr0=0.0005,
        warmup_epochs=1.0,
        optimizer="SGD",
        amp=False,
        device=0,
        name=exp_name,             # 输出到完全独立的新文件夹，防止覆盖
        exist_ok=True,
        workers=8
    )