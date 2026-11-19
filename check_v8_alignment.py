import copy
import torch
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO
from dcnv4_module import DCNv4_2D

# 1. DCNv4 动态 Patch 补丁
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


# 2. 核心：按张量形状顺序（Shape Sequential Matching）的暴力 100% 对齐函数
def sequential_shape_align(model, pretrained_path):
    print(f"\n================ 启动顺序张量物理级 100% 强行注入 ================")
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    src_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    tar_dict = model.model.state_dict()
    
    # 提取预训练权重中所有的 Parameter 张量列表
    src_items = [(k, v) for k, v in src_dict.items()]
    
    new_state_dict = {}
    matched_count = 0
    src_idx = 0
    src_len = len(src_items)

    for tar_k, tar_v in tar_dict.items():
        # 如果是 Detect 检测头，专门跳过或者手动处理，防止 Shape 碰撞
        if "cv2" in tar_k or "cv3" in tar_k:
            continue
            
        # 在 src_dict 中按顺序寻找下一个 shape 匹配的张量
        search_idx = src_idx
        found = False
        while search_idx < src_len:
            src_k, src_v = src_items[search_idx]
            if src_v.shape == tar_v.shape:
                new_state_dict[tar_k] = src_v
                matched_count += 1
                src_idx = search_idx + 1 # 移动游标
                found = True
                break
            elif "dcn" in tar_k.lower() and src_v.numel() == tar_v.numel():
                new_state_dict[tar_k] = src_v.view_as(tar_v)
                matched_count += 1
                src_idx = search_idx + 1
                found = True
                break
            search_idx += 1
            
        if not found:
            # 如果按游标没搜到，全局搜一次 Shape 完全一致的
            for global_k, global_v in src_items:
                if global_v.shape == tar_v.shape and global_k not in [k for k in new_state_dict.keys()]:
                    new_state_dict[tar_k] = global_v
                    matched_count += 1
                    break

    # 覆盖模型权重
    tar_dict.update(new_state_dict)
    model.model.load_state_dict(tar_dict, strict=False)
    
    total_tar_layers = len(tar_dict)
    print(f"✅ 成功强行匹配并注入的层数: {matched_count} / {total_tar_layers}")
    print(f"🔥 骨架权重物理覆盖率: {matched_count / total_tar_layers * 100:.2f}%\n")
    return model


if __name__ == "__main__":
    yaml_config = "yolov8-p2-dcnv4-dfire.yaml"
    exp_name = "exp2_stage2_ours_dfire_seq_aligned"
    pretrained_weights = r"F:\ultralyticsyolo8\yolov8n.pt"

    # 1. 实例化模型
    model = YOLO(yaml_config, task="detect")
    
    # 2. 执行顺序张量强行注入
    model = sequential_shape_align(model, pretrained_weights)
    
    # 3. 激活 NWD Loss
    model.args = {"use_nwd": True}

    # 4. 启动训练（观察前 3 轮）
    results = model.train(
        data="dfire.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        lr0=0.001,
        warmup_epochs=3.0,
        optimizer="SGD",
        amp=False,
        device=0,
        name=exp_name,
        exist_ok=True,
        workers=8
    )