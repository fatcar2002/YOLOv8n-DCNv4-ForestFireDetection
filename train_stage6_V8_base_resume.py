import os
import copy
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO

# 1. 导入 DCNv4 模块
try:
    from dcnv4_module import DCNv4_2D
except ImportError:
    raise ImportError("请确保 dcnv4_module.py 在当前运行目录中！")

# 🚀 强行将 DCNv4_2D 注册到 Ultralytics 核心命名空间
setattr(modules, "DCNv4_2D", DCNv4_2D)
setattr(modules.block, "DCNv4_2D", DCNv4_2D)
setattr(tasks, "DCNv4_2D", DCNv4_2D)
tasks.__dict__["DCNv4_2D"] = DCNv4_2D

# 🚀 开启 RTX 5090 D TF32 极速加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# 2. DCNv4 专属严谨 Patch
# =====================================================================
_orig_parse_model = tasks.parse_model

def patched_parse_model(d, ch, verbose=True):
    d = copy.deepcopy(d)
    for section_key in ['backbone', 'head']:
        if section_key in d:
            for i, layer in enumerate(d[section_key]):
                if len(layer) > 2 and layer[2] == 'DCNv4_2D':
                    layer[2] = 'Conv'
    model_seq, save_list = _orig_parse_model(d, ch, verbose)
    return model_seq, save_list

tasks.parse_model = patched_parse_model

# =====================================================================
# 3. 标准断点续训入口
# =====================================================================
if __name__ == '__main__':
    # 🎯 指向你上一次 Ours 实验目录下生成的 weights/last.pt
    # （例如：runs/detect/stage6_yolov8_ours_warmstart/weights/last.pt）
    last_weights = r"runs/detect/runs/detect/stage6_yolov8_ours_warmstart/weights/last.pt"
    
    if not os.path.exists(last_weights):
        raise FileNotFoundError(f"未找到断点权重文件: {last_weights}，请检查路径！")

    dataset_yaml = r"F:\ultralyticsyolo8\datasets_D-fire-pure-small\pure_fire.yaml"
    project_name = "runs/detect"
    exp_name = "stage6_yolov8_ours_warmstart"  # 必须与上一次实验的 name 完全一致

    print("================ 🚀 启动 YOLOv8 Ours 模型标准断点续训 (Resume) ================")

    # 1. 直接加载 last.pt
    model = YOLO(last_weights)

    # 2. 显式挂载 NWD 损失函数
    model.args = {"use_nwd": True}
    if hasattr(model, "model") and model.model is not None:
        model.model.args = {"use_nwd": True}

    # 3. 启动续训：直接设置 epochs 为你的目标总代数（例如 180 或 200）
    results = model.train(
        data=dataset_yaml,
        epochs=80,                 # 目标总代数（YOLO 会自动识别并从当前进度继续往后跑）
        resume=True,                # 🎯 核心断点续训开关：自动加载优化器状态、学习率衰减和历史代数
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        amp=False,
        cache=True,
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )
    
    print("\n🎉 Ours 模型断点续训成功完成！")