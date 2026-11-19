import os
import copy
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO

# 导入你的 DCNv4 模块实现
try:
    from dcnv4_module import DCNv4_2D
except ImportError:
    raise ImportError("请确保 dcnv4_module.py 在当前运行目录中！")

# 🚀 开启 RTX 5090 D 专属 TF32 极速加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# 1. DCNv4 专属严谨 Patch (强行锁死 k=3 避开 invalid kernel shape)
# =====================================================================
_orig_parse_model = tasks.parse_model

def patched_parse_model(d, ch, verbose=True):
    d = copy.deepcopy(d)
    dcn_layers = {}
    layers = d.get('backbone', []) + d.get('head', [])
    for i, layer in enumerate(layers):
        if layer[2] == 'DCNv4_2D':
            dcn_layers[i] = layer[3]
            layer[2] = 'Conv'  # 先用标准 Conv 占位
            
    model_seq, save_list = _orig_parse_model(d, ch, verbose)
    
    for i, dcn_args in dcn_layers.items():
        placeholder_conv = model_seq[i]
        
        # 🎯 鲁棒提取 in_channels 和 out_channels
        c1, c2 = None, None
        if hasattr(placeholder_conv, 'conv'):
            c1 = placeholder_conv.conv.in_channels
            c2 = placeholder_conv.conv.out_channels
        elif isinstance(placeholder_conv, nn.Sequential):
            for sub_m in placeholder_conv:
                if isinstance(sub_m, nn.Conv2d):
                    c1, c2 = sub_m.in_channels, sub_m.out_channels
                    break
        elif hasattr(placeholder_conv, 'in_channels'):
            c1 = placeholder_conv.in_channels
            c2 = placeholder_conv.out_channels

        # 如果没提取到，使用配置通道值
        if c1 is None or c2 is None:
            c1 = dcn_args[0]
            c2 = dcn_args[0]

        # 🚨 核心修复：DCNv4 C++ 算子严格要求 k=3，强行修复步进与卷积核
        k = 3  # 锁死 k=3，绝对防止传入 k=1 触发 invalid kernel shape 报错
        s = 1  # 默认 stride=1

        # 实例化 DCNv4
        new_m = DCNv4_2D(c1, c2, k=k, s=s)
        new_m.f = getattr(placeholder_conv, 'f', -1)
        new_m.i = getattr(placeholder_conv, 'i', i)
        new_m.type = 'DCNv4_2D'
        new_m.np = sum(p.numel() for p in new_m.parameters())
        model_seq[i] = new_m
        
    return model_seq, save_list

# 挂载 Patch
tasks.parse_model = patched_parse_model
setattr(modules, "DCNv4_2D", DCNv4_2D)
setattr(modules.block, "DCNv4_2D", DCNv4_2D)
setattr(tasks, "DCNv4_2D", DCNv4_2D)
tasks.__dict__["DCNv4_2D"] = DCNv4_2D

# =====================================================================
# 2. 主训练入口
# =====================================================================
if __name__ == '__main__':
    yaml_config = "yolov8-p2-dcnv4-nwd-smalltarget.yaml"
    dataset_yaml = r"F:\ultralyticsyolo8\datasets300test_small\stage3_mini300_dfire.yaml"
    pretrained_weights = r"F:\ultralyticsyolo8\yolov8n.pt"

    project_name = "runs/detect"
    exp_name = "exp_mini300_ours_p2_dcnv4_nwd_full"

    print("================ 🚀 启动终极完全体: YOLOv8n + P2 + DCNv4 + NWD ================")

    # 1. 加载 YAML 模型结构，并自动交集继承权重[cite: 7]
    model = YOLO(yaml_config, task="detect").load(pretrained_weights)

    # 2. 挂载 NWD 损失函数
    model.args = {"use_nwd": True}
    if hasattr(model, "model") and model.model is not None:
        model.model.args = {"use_nwd": True}

    # 3. 启动训练
    results = model.train(
        data=dataset_yaml,
        epochs=120,
        imgsz=640,
        batch=16,
        workers=16,
        device=0,
        amp=False,                 # 保持全精度 FP32/TF32，防止 NWD/DCN 梯度下溢
        cache=False,
        
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.05,
        
        warmup_epochs=5.0,         # 5 轮预热
        cos_lr=True,
        close_mosaic=10,
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    print("\n🎉 完全体模型训练完毕！")