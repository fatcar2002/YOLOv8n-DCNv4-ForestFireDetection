import os
import copy
import torch
from pathlib import Path
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO
from dcnv4_module import DCNv4_2D

# 🚀 开启 RTX 5090 D TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# 1. 完全复刻 train_stage2_ours_dfire.py 的 DCNv4 注册 Patch (支持 s=2 下采样)
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
# 2. YOLOv8 Ours 批量推理主入口
# =====================================================================
if __name__ == '__main__':
    # YOLOv8 Ours 权重路径
    WEIGHT_PATH = r"F:\ultralyticsyolo8\runs\detect\exp2_stage2_ours_dfire\weights\best.pt"
    PROJECT_DIR = r"F:\ultralyticsyolo8"
    # 指定保存在 inference_resultsv8 文件夹内
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "inference_resultsv8")

    # 保持与之前完全一致的三个测试集路径
    TEST_DATASETS = {
        # "night_test": r"F:\ultralyticsyolo8\dataset_night_test\images",
        # "small_D_fire_test": r"F:\ultralyticsyolo8\datasets_stage3_small__D_fire",
        # "Ue_middlesmall_test": r"F:\ultralyticsyolo8\datasets_Ue_middlesmall_target"
        "OHID_test": r"F:\ultralyticsyolo8\dataset_stage1_ohid\images"
    }

    print(f"📡 正在载入 YOLOv8-Ours 权重: {WEIGHT_PATH}")
    model = YOLO(WEIGHT_PATH)
    model.args = {"use_nwd": True}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, data_path in TEST_DATASETS.items():
        print(f"\n⚡ 正在推理数据集: {name}")
        
        # 自动定位测试集图片源目录
        source_path = data_path
        for sub_dir in [os.path.join(data_path, "test", "images"), 
                        os.path.join(data_path, "images"), 
                        os.path.join(data_path, "test")]:
            if os.path.exists(sub_dir) and any(Path(sub_dir).glob("*.*")):
                source_path = sub_dir
                break
                
        print(f"📸 图片源目录: {source_path}")

        # 执行预测（与训练 amp=False 对齐，采用 half=False 全精度前向推理）
        results = model.predict(
            source=source_path,
            save=True,             # 保存带 Ultralytics 风格检测框的图片
            save_txt=True,         # 保存预测框 txt 坐标备用
            conf=0.25,             # 论文主评估基准阈值 0.25（如需抓极微弱火点可改 0.15）
            iou=0.45,
            imgsz=640,
            half=False,
            project=OUTPUT_DIR,    # 输出总目录：F:\ultralyticsyolo8\inference_resultsv8
            name=f"{name}_results",
            exist_ok=True
        )

        save_dir = results[0].save_dir if results else "未知"
        print(f"🎉 {name} 推理完成！结果已保存至: {save_dir}")

    print("\n✨ YOLOv8-Ours 全部测试集推理完成！")