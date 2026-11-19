import os
import copy
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO

# 导入 DCNv4 模块
try:
    from dcnv4_module import DCNv4_2D
except ImportError:
    raise ImportError("请确保 dcnv4_module.py 在当前运行目录中！")

# 🚀 开启 RTX 5090 D TF32 极速加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# 1. DCNv4 专属严谨 Patch (锁死 k=3 避开 C++ 算子错误)
# =====================================================================
_orig_parse_model = tasks.parse_model

def patched_parse_model(d, ch, verbose=True):
    d = copy.deepcopy(d)
    dcn_layers = {}
    layers = d.get('backbone', []) + d.get('head', [])
    for i, layer in enumerate(layers):
        if layer[2] == 'DCNv4_2D':
            dcn_layers[i] = layer[3]
            layer[2] = 'Conv'
            
    model_seq, save_list = _orig_parse_model(d, ch, verbose)
    
    for i, dcn_args in dcn_layers.items():
        placeholder_conv = model_seq[i]
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

        if c1 is None or c2 is None:
            c1 = dcn_args[0]
            c2 = dcn_args[0]

        k, s = 3, 1
        new_m = DCNv4_2D(c1, c2, k=k, s=s)
        new_m.f = getattr(placeholder_conv, 'f', -1)
        new_m.i = getattr(placeholder_conv, 'i', i)
        new_m.type = 'DCNv4_2D'
        new_m.np = sum(p.numel() for p in new_m.parameters())
        model_seq[i] = new_m
        
    return model_seq, save_list

tasks.parse_model = patched_parse_model
setattr(modules, "DCNv4_2D", DCNv4_2D)
setattr(modules.block, "DCNv4_2D", DCNv4_2D)
setattr(tasks, "DCNv4_2D", DCNv4_2D)
tasks.__dict__["DCNv4_2D"] = DCNv4_2D

# =====================================================================
# 2. 训练主入口
# =====================================================================
if __name__ == '__main__':
    yaml_config = "yolo26-p2-dcnv4-dfire-small.yaml"
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_stage4_micro_fire_only\micro_fire_only.yaml"
    pretrained_weights = r"F:\ultralyticsyolo8\weights\yolo26n.pt"  # 确认预训练权重路径存在

    project_name = "runs/detect"
    exp_name = "exp_yolo26_ours_pretrained_fast70"

    print("================ 🚀 启动恢复预训练继承版 YOLOv26 Ours (快攻 70%+) ================")

    # 1. 实例化模型并精准继承 yolo26n.pt 权重
    model = YOLO(yaml_config, task="detect").load(pretrained_weights)

    # 2. 挂载 NWD 损失函数
    model.args = {"use_nwd": True}
    if hasattr(model, "model") and model.model is not None:
        model.model.args = {"use_nwd": True}

    # 3. 启动训练 (跑 120 轮以保障充分收敛)
    results = model.train(
        data=dataset_yaml,
        epochs=120,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        amp=False,
        cache=True,
        optimizer="AdamW",     # 预训练模式下配合 AdamW 收敛极快
        lr0=0.001,
        warmup_epochs=5.0,
        cos_lr=True,
        close_mosaic=10,
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    print("\n🎉 YOLOv26 Ours 快攻版训练完成！")