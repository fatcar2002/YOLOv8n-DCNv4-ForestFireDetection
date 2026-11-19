import os
import copy
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules

# 🚀 开启 RTX 5090 D TF32 极速推理加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# 1. 导入 DCNv4 模块并挂载自定义算子（必须保留，否则报错无法加载 DCNv4）
# =====================================================================
try:
    from dcnv4_module import DCNv4_2D
except ImportError:
    raise ImportError("🚨 请确保 dcnv4_module.py 以及其对应的编译文件在当前运行目录中！")

# DCNv4 专属严谨 Patch (锁死 k=3 避开 C++ 算子错误，确保推理与训练拓扑 100% 一致)
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

# 动态打补丁，使 YOLO 在加载权重时能够识别 DCNv4_2D 层
tasks.parse_model = patched_parse_model
setattr(modules, "DCNv4_2D", DCNv4_2D)
setattr(modules.block, "DCNv4_2D", DCNv4_2D)
setattr(tasks, "DCNv4_2D", DCNv4_2D)
tasks.__dict__["DCNv4_2D"] = DCNv4_2D


# =====================================================================
# 2. 定性推理主入口
# =====================================================================
if __name__ == '__main__':
    # 核心路径配置
    WEIGHT_PATH = r"F:\ultralyticsyolo8\runs\detect\runs\detect\exp_stage6_yolo26_ours_640px_push78_final\weights\best.pt"
    PROJECT_DIR = r"F:\ultralyticsyolo8"
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "inference_results2") # 自动新建的推理结果总目录

    # 待测试的三个数据集/测试集路径
    TEST_DATASETS = {
        "night_test": r"F:\ultralyticsyolo8\dataset_night_test\images",
        "small_D_fire_test": r"F:\ultralyticsyolo8\datasets_stage3_small__D_fire",
        "Ue_middlesmall_test": r"F:\ultralyticsyolo8\datasets_Ue_middlesmall_target"
    }

    print("================ 🕵️ 启动 YOLOv8/26 Ours 批量推理 =================\n")
    print(f"📡 正在载入自定义模型权重并应用 DCNv4 补丁: {WEIGHT_PATH}")
    
    # 1. 挂载 Patch 并加载模型
    model = YOLO(WEIGHT_PATH)
    
    # 2. 注入 NWD 回归损失相关设置，保持与训练时对小目标尺度的微调策略对齐
    model.args = {"use_nwd": True}
    if hasattr(model, "model") and model.model is not None:
        model.model.args = {"use_nwd": True}

    # 创建结果总目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. 循环遍历测试集推理并保存
    for name, data_path in TEST_DATASETS.items():
        print(f"\n⚡ 当前推理数据集: {name}")
        
        # 自动定位图片真实存放目录（防止空文件夹情况）
        source_path = data_path
        for sub_dir in [os.path.join(data_path, "test", "images"), 
                        os.path.join(data_path, "images"), 
                        os.path.join(data_path, "test")]:
            if os.path.exists(sub_dir) and any(Path(sub_dir).glob("*.*")):
                source_path = sub_dir
                break
                
        print(f"📸 确定图片源目录: {source_path}")

        # 4. 执行预测
        # conf=0.15: 调低至 0.15 可以极大释放夜间、极微小场景下的低对比度火点与淡烟雾 (smoke & fire)
        # conf=0.25: 也是可用的，不过对于大雾/夜间漏检的补救，低置信度可以提供更敏感的捕捉
        results = model.predict(
            source=source_path,
            save=True,             # 核心：保存标框图（Ultralytics 风格）
            save_txt=True,         # 额外保存 txt labels 备用
            conf=0.15,             # 🔥 推荐降低到 0.15 - 0.20 提升微弱火、烟的边界召回
            iou=0.45,              # NMS 重叠抑制阈值
            imgsz=640,             # 保持与 640px 训练尺度一致
            project=OUTPUT_DIR,    # 推理结果汇总在 inference_results 文件夹
            name=f"{name}_results",# 子目录：night_test_results 等
            exist_ok=True          # 同名时覆盖不报错
        )

        # 获取实际保存路径
        save_dir = results[0].save_dir if results else "未知"
        print(f"🎉 {name} 批量标注完毕！检测结果已落盘: {save_dir}")

    print("\n================== ✨ 3组测试集 Ultralytics 风格标框推理已全部完成！ ==================")