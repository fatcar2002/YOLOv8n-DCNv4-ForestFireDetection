import os
import copy
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO

# 1. 导入 DCNv4 模块
try:
    from dcnv4_module import DCNv4_2D
except ImportError:
    raise ImportError("请确保 dcnv4_module.py 在当前运行目录中！")

# 🚀 强行将 DCNv4_2D 注册到 Ultralytics 核心命名空间
setattr(modules, "DCNv4_2D", DCNv4_2D)
setattr(modules.block, "DCNv4_2D", DCNv4_2D)
setattr(tasks, "DCNv4_2D", DCNv4_2D)
tasks.__dict__["DCNv4_2D"] = DCNv4_2D

# 🚀 开启 RTX 5090 D TF32 极速加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# 2. DCNv4 专属 Patch (拦截模型解析并替换算子)
# =====================================================================
_orig_parse_model = tasks.parse_model

def patched_parse_model(d, ch, verbose=True):
    d = copy.deepcopy(d)
    for section_key in ['backbone', 'head']:
        if section_key in d:
            for i, layer in enumerate(d[section_key]):
                if len(layer) > 2 and layer[2] == 'DCNv4_2D':
                    layer[2] = 'Conv'
    model_seq, save_list = _orig_parse_model(d, ch, verbose)
    return model_seq, save_list

tasks.parse_model = patched_parse_model

# =====================================================================
# 3. 主训练入口
# =====================================================================
if __name__ == '__main__':
    # 🎯 核心修改：直接加载你的 YOLOv26 预训练/基础权重文件
    model_path = r"weights\yolo26n.pt"
    
    # 🎯 指向数据集配置文件路径
    dataset_yaml = r"F:\ultralyticsyolo8\datasets_D-fire-pure-small\pure_fire.yaml"
    
    project_name = "runs/detect"
    exp_name = "stage6_yolo26_ours_2+1small"  # 🎯 对应修改实验命名

    print("================ 🚀 启动 YOLOv26 Ours 魔改版高性能训练 ================")

    # 1. 直接通过权重实例化模型（完美适配 yolo26n.pt 内部结构）
    model = YOLO(model_path)

    # 2. 显式挂载 NWD 损失函数
    model.args = {"use_nwd": True}
    if hasattr(model, "model") and model.model is not None:
        model.model.args = {"use_nwd": True}

    # 3. 启动 100 代微调训练
    results = model.train(
        data=dataset_yaml,
        epochs=260,                 
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        amp=False,
        cache=True,
        optimizer="AdamW",          
        lr0=0.0005,                 # 建议微调时用稍小的学习率
        cos_lr=True,                
        close_mosaic=10,            
        
        # 🎯 小目标专属调优超参数
        box=9.5,               
        cls=0.6,               
        scale=0.7,             
        copy_paste=0.1,        
        
        project=project_name,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    print("\n🎉 YOLOv26 Ours 训练顺利完成！")