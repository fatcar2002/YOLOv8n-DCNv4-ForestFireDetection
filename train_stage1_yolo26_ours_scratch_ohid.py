import copy
import torch
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO
from dcnv4_module import DCNv4_2D

# 1. DCNv4 动态 Patch (确保自定义 DCNv4 算子在纯 YAML 加载时正确注册)
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
        c1 = placeholder_conv.conv.in_channels
        c2 = placeholder_conv.conv.out_channels
        k = dcn_args[1] if len(dcn_args) > 1 else 3
        s = dcn_args[2] if len(dcn_args) > 2 else 1
        
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


if __name__ == "__main__":
    # 使用针对遥感单类烟雾配置的 Ours YAML 配置文件 (nc: 1)
    yaml_config = "yolo26-p2-dcnv4-ohid.yaml"
    exp_name = "exp1_stage1_yolo26_ours_scratch_ohid"

    # 1. 仅通过 YAML 实例化模型（0 权重全新初始化，断开所有预训练 .pt 载入）
    model = YOLO(yaml_config, task="detect")
    
    # 2. 显式激活 NWD Loss
    model.args = {"use_nwd": True}

    # 3. 启动遥感数据集（OHID-FF）150 轮训练
    results = model.train(
        data="ohid_ff.yaml",        # 遥感数据集配置文件[cite: 13]
        epochs=150,                 # 150 轮[cite: 13]
        imgsz=640,                  # 640x640[cite: 13]
        batch=8,                    # 保持 batch=8 严格控制变量[cite: 13]
        lr0=0.0005,                 # 学习率 0.0005[cite: 13]
        warmup_epochs=10.0,         # 10 轮预热[cite: 13]
        optimizer="SGD",            # SGD 优化器[cite: 13]
        amp=False,                  # 关闭混合精度[cite: 13]
        device=0,                   # 指定显卡 0[cite: 13]
        name=exp_name,              # 实验命名
        exist_ok=True,              # 允许同名覆盖[cite: 13]
        workers=8                   # 8 个 CPU 线程[cite: 13]
    )