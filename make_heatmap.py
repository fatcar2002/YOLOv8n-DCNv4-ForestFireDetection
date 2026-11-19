import os
import sys
import copy
import types
import cv2
import numpy as np
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================================================================
# 1. SafeDCNv4 结构补丁 (彻底规避 C++ ext.pyd DLL 报错)
# =====================================================================
class SafeDCNv4(nn.Module):
    def __init__(self, c1=64, c2=64, k=3, s=1, *args, **kwargs):
        super().__init__()
        if len(args) >= 1: c1 = args[0]
        if len(args) >= 2: c2 = args[1]
        self.in_channels = c1
        self.out_channels = c2
        self.stride = s
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x, *args, **kwargs):
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.shape[1] != self.conv.in_channels:
            in_c = x.shape[1]
            out_c = self.out_channels
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=self.stride, padding=1, bias=False).to(x.device)
            self.bn = nn.BatchNorm2d(out_c).to(x.device)
            self.in_channels = in_c
        return self.act(self.bn(self.conv(x)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)

dummy_module = types.ModuleType("DCNv4")
dummy_module.DCNv4 = SafeDCNv4
dummy_module.DCNv4_2D = SafeDCNv4

dummy_modules_sub = types.ModuleType("DCNv4.modules")
dummy_modules_sub.dcnv4 = dummy_module
dummy_modules_sub.DCNv4 = SafeDCNv4

dummy_dcnv4_mod = types.ModuleType("dcnv4_module")
dummy_dcnv4_mod.DCNv4_2D = SafeDCNv4
dummy_dcnv4_mod.DCNv4 = SafeDCNv4

sys.modules["DCNv4"] = dummy_module
sys.modules["DCNv4.modules"] = dummy_modules_sub
sys.modules["DCNv4.modules.dcnv4"] = dummy_module
sys.modules["dcnv4_module"] = dummy_dcnv4_mod

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO

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
        new_m = SafeDCNv4(c1, c2, k=k, s=s)
        new_m.f = getattr(placeholder_conv, 'f', -1)
        new_m.i = getattr(placeholder_conv, 'i', i)
        new_m.type = 'DCNv4_2D'
        model_seq[i] = new_m
    return model_seq, save_list

tasks.parse_model = patched_parse_model
setattr(modules, "DCNv4_2D", SafeDCNv4)
setattr(modules.block, "DCNv4_2D", SafeDCNv4)
setattr(tasks, "DCNv4_2D", SafeDCNv4)
tasks.__dict__["DCNv4_2D"] = SafeDCNv4

def safe_patch_dcn_meta(model_seq):
    for idx, layer in enumerate(model_seq):
        layer_name = layer.__class__.__name__
        if 'DCNv4' in layer_name or layer_name == 'SafeDCNv4':
            orig_f = getattr(layer, 'f', -1)
            orig_i = getattr(layer, 'i', idx)
            orig_type = getattr(layer, 'type', 'DCNv4_2D')
            if idx == 5: in_c, out_c = 256, 512
            elif idx == 7: in_c, out_c = 512, 1024
            else:
                in_c = getattr(layer, 'in_channels', 256)
                out_c = getattr(layer, 'out_channels', 512)
            new_layer = SafeDCNv4(in_c, out_c, k=3, s=2)
            new_layer.f = orig_f
            new_layer.i = orig_i
            new_layer.type = orig_type
            model_seq[idx] = new_layer

# =====================================================================
# 2. 自动检索小目标图片
# =====================================================================
def find_smallest_target_images(img_dir, lbl_dir, top_k=15):
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    img_area_list = []

    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            min_area_in_img = 1.0
            valid = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    w, h = float(parts[3]), float(parts[4])
                    area = w * h
                    if area > 0:
                        min_area_in_img = min(min_area_in_img, area)
                        valid = True
            
            if valid:
                img_area_list.append((img_name, min_area_in_img))

    img_area_list.sort(key=lambda x: x[1])
    selected_imgs = [item[0] for item in img_area_list[:top_k]]
    return selected_imgs

# =====================================================================
# 3. 稳健的纯前向 Hook 热力图提取器
# =====================================================================
def extract_single_heatmap(yolo_model, img_path, target_layer_idx):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w, _ = img.shape

    # 预处理图片为标准 Tensor
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    fmap_container = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            fmap_container.append(output.detach().cpu())
        elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            fmap_container.append(output[0].detach().cpu())

    # 获取目标特征层
    try:
        target_layer = yolo_model.model.model[target_layer_idx]
    except (IndexError, AttributeError):
        target_layer = yolo_model.model.model[-3]

    # 挂载 Hook 并直接调用底层的 forward_once，避免 predict() 内部状态污染
    handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        try:
            # 直接触发网络前向提取
            _ = yolo_model.model.forward_once(img_tensor)
        except Exception:
            # 容错降级：尝试全网络前向
            try:
                _ = yolo_model.model(img_tensor)
            except Exception:
                pass

    handle.remove()

    if len(fmap_container) > 0:
        # 提取最后一层输出特征图
        fmap = fmap_container[-1][0].numpy()
        heatmap = np.mean(fmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        return overlay

    return None

# =====================================================================
# 4. 主程序
# =====================================================================
if __name__ == '__main__':
    baseline_weight = "runs/detect/exp2_stage2_baseline_dfire/weights/best.pt"
    ours_weight = "runs/detect/exp2_stage2_ours_dfire/weights/best.pt"
    
    img_dir = r"datasets_stage2_D-fire\test\images"
    lbl_dir = r"datasets_stage2_D-fire\test\labels"
    output_dir = r"F:\ultralyticsyolo8\heatmap_results"
    os.makedirs(output_dir, exist_ok=True)

    print("🔍 正在扫描 D-Fire 测试集，按目标框面积自动寻找极小目标样本...")
    small_target_imgs = find_smallest_target_images(img_dir, lbl_dir, top_k=15)
    print(f"🎯 已锁定最典型的 15 张微小目标图片: {small_target_imgs}")

    print("\n================ 开始批量生成热力图 ================")
    print(f"正在加载 Baseline 模型: {baseline_weight} ...")
    model_base = YOLO(baseline_weight)
    model_base.model.to(device)
    model_base.model.eval()
    
    print(f"正在加载 Ours 完全体模型: {ours_weight} ...")
    model_ours = YOLO(ours_weight)
    safe_patch_dcn_meta(model_ours.model.model)
    model_ours.model.to(device)
    model_ours.model.eval()

    success_count = 0
    for img_name in small_target_imgs:
        img_path = os.path.join(img_dir, img_name)
        base_filename = os.path.splitext(img_name)[0]
        
        # Baseline (P3 15层) vs Ours (P2 18层)
        hm_base = extract_single_heatmap(model_base, img_path, target_layer_idx=15)
        hm_ours = extract_single_heatmap(model_ours, img_path, target_layer_idx=18)

        if hm_base is not None and hm_ours is not None:
            combined = np.hstack((hm_base, hm_ours))
            save_combine_path = os.path.join(output_dir, f"{base_filename}_Compare.jpg")
            cv2.imwrite(save_combine_path, combined)
            print(f"✅ [成功导出热力图对比大图]: {base_filename}_Compare.jpg")
            success_count += 1
        else:
            print(f"⚠️ [导出提示]: {base_filename} 提取特征失败")

    print(f"\n🎉 检索与提取完成！成功导出 {success_count}/15 张对比图。请打开文件夹挑选：\n{os.path.abspath(output_dir)}")