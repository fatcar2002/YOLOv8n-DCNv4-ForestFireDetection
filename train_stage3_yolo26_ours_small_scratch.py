import copy
import torch
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO
from dcnv4_module import DCNv4_2D

# 🚀 开启 TF32 模式：全面激活 RTX 5090 D 的 Tensor Core 硬件加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# 1. DCNv4 动态 Patch 补丁 (解析 YAML 中的 DCNv4_2D 算子)
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


# =====================================================================
# 2. YOLO26 Ours (P2+DCNv4+NWD) 0 权重 Scratch 训练流程
# =====================================================================
if __name__ == "__main__":
    # YOLO26 + P2 + DCNv4 配置文件名（请确保文件名与本地 yaml 完全一致）
    yaml_config = "yolo26-p2-dcnv4-dfire.yaml" 
    
    # 指向 Stage 3 专属 1000 张小目标数据集
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_stage3_small__D_fire\stage3_small_dfire.yaml"

    project_name = "runs/detect"
    exp_name = "exp3_stage3_yolo26_ours_small_scratch"

    print("================ 🚀 开始运行 Stage 3: YOLO26 Ours (P2+DCNv4+NWD, 0权重/Scratch) ================")
    
    # 1. 仅加载 YAML 实例化模型（0 权重全新随机初始化，不载入任何 .pt 预训练权重）
    model = YOLO(yaml_config, task="detect")
    
    # 2. 挂载 NWD 损失函数
    model.args = {"use_nwd": True}

    # 3. 启动训练（应用实测最佳 Windows 稳定参数）
    results = model.train(
        data=dataset_yaml,
        epochs=150,
        imgsz=640,
        batch=16,                 # 🚀 设为 16，保证显存安全且计算高效
        workers=0,                # 🚀 必须为 0，彻底防止 Windows 多进程死锁卡死
        device=0,
        amp=False,                # 🚀 保持 False (FP32/TF32)，防止 DCNv4+NWD 在 FP16 下梯度 NaN
        cache=False,              # 🚀 保持 False，防止物理内存溢出引发系统假死
        lr0=0.0005,               # 保持 YOLO26 推荐的 0.0005 Scratch 初始学习率
        warmup_epochs=10.0,
        optimizer="SGD",
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )