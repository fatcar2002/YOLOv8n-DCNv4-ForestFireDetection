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
# 2. 主训练入口 (完全摒弃有冲突的预训练权重加载，从干净结构起步)
# =====================================================================
if __name__ == '__main__':
    # 🎯 确保这里的 YAML 里面的 nc: 2 已经与双类 UE 数据集完美对应
    yaml_config = "yolov8-p2-dcnv4-nwd-dcnv4-smalltarget.yaml"
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_Ue_middlesmall_target\UEdataset.yaml"

    project_name = "runs/detect"
    exp_name = "stage5_yolov8_UE_ours_pure_scratch"

    print("================ 🚀 启动 YOLOv8 Ours 纯净结构训练 (避免矩阵维度冲突) ================")

    # 1. 直接从 YAML 实例化模型（不调用 .load() 任何异构权重，彻底根治矩阵乘法报错）
    model = YOLO(yaml_config, task="detect")

    # 2. 挂载 NWD 损失函数
    model.args = {"use_nwd": True}
    if hasattr(model, "model") and model.model is not None:
        model.model.args = {"use_nwd": True}

    # 3. 启动 100 代训练
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        amp=False,             # 保持全精度 FP32/TF32 防止 NWD/DCN 下溢
        cache=True,
        
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.05,
        warmup_epochs=5.0,
        cos_lr=True,
        
        # 🎯 核心调优超参数
        box=9.5,               # 压实微小目标定位
        cls=0.6,               # 平衡分类权重
        scale=0.7,             # 触发 P2 浅层高分辨率特征响应
        copy_paste=0.1,        # 增强小目标密度
        close_mosaic=10,       # 最后 10 代关闭 Mosaic
        
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    print("\n🎉 YOLOv8 Ours 纯净结构训练顺利完成！")