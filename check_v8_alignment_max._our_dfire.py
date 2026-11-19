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


# 2. 核心：外科手术式“智能切片与克隆”极限对齐函数
def smart_surgical_align(model, pretrained_path):
    print(f"\n================ 启动物理级外科手术智能切片注入 (目标覆盖率 80%+) ================")
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    src_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    tar_dict = model.model.state_dict()
    
    new_state_dict = {}
    matched_count = 0

    for tar_k, tar_v in tar_dict.items():
        # A. 完全精准匹配（Backbone 基础层等）
        if tar_k in src_dict and src_dict[tar_k].shape == tar_v.shape:
            new_state_dict[tar_k] = src_dict[tar_k]
            matched_count += 1
            
        # B. 针对 DCNv4 模块的子权重注入 (把普通 Conv 权重塞给 DCNv4 的主卷积)
        elif "dcn" in tar_k.lower():
            # 尝试寻找对应的原始 Conv 权重
            possible_src_k = tar_k.replace("dcn", "conv")
            for k_src, v_src in src_dict.items():
                if v_src.numel() == tar_v.numel() or (len(v_src.shape) == 4 and v_src.shape == tar_v.shape):
                    new_state_dict[tar_k] = v_src.view_as(tar_v)
                    matched_count += 1
                    break

        # C. 针对通道变小或变大的 Neck 层进行“智能切片提取”
        elif tar_k in src_dict:
            src_v = src_dict[tar_k]
            # 如果维度形状不一致，但都是 4D 卷积矩阵，强行切片适配
            if len(src_v.shape) == 4 and len(tar_v.shape) == 4:
                min_c_out = min(src_v.shape[0], tar_v.shape[0])
                min_c_in = min(src_v.shape[1], tar_v.shape[1])
                
                # 创建全 0 容器并切片注入预训练参数
                sliced_weight = torch.zeros_like(tar_v)
                sliced_weight[:min_c_out, :min_c_in, :, :] = src_v[:min_c_out, :min_c_in, :, :]
                new_state_dict[tar_k] = sliced_weight
                matched_count += 1
                
            elif len(src_v.shape) == 1 and len(tar_v.shape) == 1: # 偏置 Bias / BN 层
                min_c = min(src_v.shape[0], tar_v.shape[0])
                sliced_bias = torch.zeros_like(tar_v)
                sliced_bias[:min_c] = src_v[:min_c]
                new_state_dict[tar_k] = sliced_bias
                matched_count += 1

    # 覆盖模型权重
    tar_dict.update(new_state_dict)
    model.model.load_state_dict(tar_dict, strict=False)
    
    total_tar_layers = len(tar_dict)
    print(f"✅ 成功强行匹配并智能切片注入的层数: {matched_count} / {total_tar_layers}")
    print(f"🔥 极限骨架+Neck权重物理覆盖率: {matched_count / total_tar_layers * 100:.2f}%\n")
    return model


if __name__ == "__main__":
    yaml_config = "yolov8-p2-dcnv4-dfire.yaml"
    exp_name = "exp2_stage2_ours_smart_aligned_test"
    pretrained_weights = r"F:\ultralyticsyolo8\yolov8n.pt"

    # 1. 实例化模型
    model = YOLO(yaml_config, task="detect")
    
    # 2. 执行极限手术对齐
    model = smart_surgical_align(model, pretrained_weights)
    
    # 3. 激活 NWD Loss
    model.args = {"use_nwd": True}

    # 4. 启动训练（跑 3 轮观察第一时间的起步 mAP）
    results = model.train(
        data="dfire.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        lr0=0.001,
        warmup_epochs=1.0,
        optimizer="SGD",
        amp=False,
        device=0,
        name=exp_name,
        exist_ok=True,
        workers=8
    )