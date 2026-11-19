import copy
import torch
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO
from dcnv4_module import DCNv4_2D

# 1. 动态拓扑解析补丁（仅在本脚本内生效，用于注册 DCNv4_2D）
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

# 2. 终极完全体训练入口
if __name__ == "__main__":
    yaml_config = "yolov8-p2-dcnv4-ohid.yaml"
    exp_name = "exp1_stage1_ours_ohid"

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