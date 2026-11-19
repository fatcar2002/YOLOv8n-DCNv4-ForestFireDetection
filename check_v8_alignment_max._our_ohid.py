import copy
import torch
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO
from dcnv4_module import DCNv4_2D

# 1. DCNv4 动态 Patch 补丁（确保模型能正确解析自定义 DCNv4_2D 算子）
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


# 2. 核心：外科手术式“智能切片与克隆”极限对齐函数 (保持 ~35% 覆盖率注入)
def smart_surgical_align(model, pretrained_path):
    print(f"\n================ 启动遥感数据集(OHID-FF) 外科手术智能切片对齐 (目标 ~35% 覆盖率) ================")
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
            
        # B. 针对 DCNv4 模块的子权重注入
        elif "dcn" in tar_k.lower():
            for k_src, v_src in src_dict.items():
                if v_src.numel() == tar_v.numel() or (len(v_src.shape) == 4 and v_src.shape == tar_v.shape):
                    new_state_dict[tar_k] = v_src.view_as(tar_v)
                    matched_count += 1
                    break

        # C. 针对通道变动或重构的 Neck 层进行“智能切片提取”
        elif tar_k in src_dict:
            src_v = src_dict[tar_k]
            if len(src_v.shape) == 4 and len(tar_v.shape) == 4:
                min_c_out = min(src_v.shape[0], tar_v.shape[0])
                min_c_in = min(src_v.shape[1], tar_v.shape[1])
                
                sliced_weight = torch.zeros_like(tar_v)
                sliced_weight[:min_c_out, :min_c_in, :, :] = src_v[:min_c_out, :min_c_in, :, :]
                new_state_dict[tar_k] = sliced_weight
                matched_count += 1
                
            elif len(src_v.shape) == 1 and len(tar_v.shape) == 1:
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
    print(f"🔥 遥感模型骨架+Neck权重物理覆盖率: {matched_count / total_tar_layers * 100:.2f}%\n")
    return model


if __name__ == "__main__":
    # 3. 遥感模型配置与路径
    yaml_config = "yolov8-p2-dcnv4-ohid.yaml"       # 遥感专用的 P2+DCNv4 结构文件[cite: 4]
    dataset_yaml = "ohid_ff.yaml"                    # 遥感数据集配置文件位置[cite: 4]
    pretrained_weights = r"F:\ultralyticsyolo8\yolov8n.pt"  # 官方预训练权重路径
    
    # 更改为你指定的实验名称
    exp_name = "check_v8_alignment_max._our_ohid"   

    # 4. 实例化模型
    model = YOLO(yaml_config, task="detect")
    
    # 5. 执行外科手术式智能切片对齐 (~35% 覆盖率)
    model = smart_surgical_align(model, pretrained_weights)
    
    # 6. 显式激活 NWD Loss
    model.args = {"use_nwd": True}

    # 7. 启动正式的遥感数据集 150 轮训练
    results = model.train(
        data=dataset_yaml,
        epochs=150,
        imgsz=640,
        batch=8,
        lr0=0.0005,          # 延续遥感实验稳定的初始学习率[cite: 4]
        warmup_epochs=10.0,  # 10轮 Warmup 保证遥感初期梯度平稳[cite: 4]
        optimizer="SGD",
        amp=False,
        device=0,
        name=exp_name,
        exist_ok=True,
        workers=8
    )