import os
import copy
import torch
import pandas as pd
from ultralytics import YOLO

# 1. 使用原生逻辑修复 parse_model，决不破坏特征图尺寸与通道
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from dcnv4_module import DCNv4_2D

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

# 2. 精准评估每个改进路径 (带 YAML 结构重建保护)
def validate_single_model(weight_path, data_yaml, is_ours=False, ours_yaml=""):
    if not os.path.exists(weight_path):
        return None
        
    # 如果是带有 P2/DCNv4 的结构，通过 YAML 重构正确的网络拓扑[cite: 11, 14]
    if is_ours and os.path.exists(ours_yaml):
        model = YOLO(ours_yaml)
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt
        model.model.load_state_dict(state_dict, strict=False)
    else:
        model = YOLO(weight_path)
        
    metrics = model.val(
        data=data_yaml,
        split='test',
        save_json=False,  # 先关闭 json，防止因为 NMS 错乱报错
        batch=8,
        imgsz=640,
        workers=0,
        verbose=False
    )
    return {
        "Precision (%)": f"{metrics.box.mp * 100:.2f}",
        "Recall (%)": f"{metrics.box.mr * 100:.2f}",
        "mAP50 (%)": f"{metrics.box.map50 * 100:.2f}",
        "mAP50-95 (%)": f"{metrics.box.map * 100:.2f}"
    }

if __name__ == '__main__':
    print("================ 🚀 修复特征图对齐重新评估 ================")
    
    # 针对之前跑出来的几组关键改进权重单独重新读取验证
    test_targets = [
        ("v8 Ours (OHID-FF 遥感)", "runs/detect/exp1_stage1_ours_ohid_DCNV4/weights/best.pt", "ohid_ff.yaml", True, "yolov8-p2-dcnv4-ohid.yaml"),
        ("v26 Ours (D-Fire 有权重对齐)", "runs/detect/exp2_stage2_yolo26_ours_aligned_dfire/weights/best.pt", "dfire.yaml", True, "yolo26-p2-dcnv4-dfire.yaml"),
        ("v8 Ours (D-Fire 57% 极限切片)", "runs/detect/exp2_stage2_ours_smart_aligned_test/weights/best.pt", "dfire.yaml", True, "yolov8-p2-dcnv4-dfire.yaml"),
        ("v26 Ours (D-Fire 0权重 Scratch)", "runs/detect/exp2_stage2_yolo26_ours_scratch_dfire/weights/best.pt", "dfire.yaml", True, "yolo26-p2-dcnv4-dfire.yaml")
    ]
    
    for name, w_path, data_y, is_o, o_yaml in test_targets:
        res = validate_single_model(w_path, data_y, is_o, o_yaml)
        if res:
            print(f"✅ [{name}] -> mAP50: {res['mAP50 (%)']}%, mAP50-95: {res['mAP50-95 (%)']}%")
        else:
            print(f"⚠️ [{name}] 路径缺失")