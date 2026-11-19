import os
import cv2
import glob
import random
import shutil

# =====================================================================
# 路径配置 (请确认你本地 OHID-FF 的实际 train 路径)
# =====================================================================
src_img_dir = r"F:\ultralyticsyolo8\dataset_stage1_ohid\images\train"
src_lbl_dir = r"F:\ultralyticsyolo8\dataset_stage1_ohid\labels\train"

# 安全导出的新路径 (独立保存，原数据集不受丝毫影响)
out_img_dir = r"F:\ultralyticsyolo8\dataset_stage1_ohid\images\train_copy_paste"
out_lbl_dir = r"F:\ultralyticsyolo8\dataset_stage1_ohid\labels\train_copy_paste"

def safe_copy_paste_augmentation(target_count=180):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print("🛡️ 正在安全复制原始数据到扩增新目录...")
    # 1. 复制原数据到新目录
    for img_p in glob.glob(os.path.join(src_img_dir, "*.*")):
        shutil.copy(img_p, out_img_dir)
    for lbl_p in glob.glob(os.path.join(src_lbl_dir, "*.txt")):
        shutil.copy(lbl_p, out_lbl_dir)

    print("✅ 原始数据已完整备份到新目录！开始提取小目标切片...")

    # 2. 从原数据集提取切片
    img_files = glob.glob(os.path.join(out_img_dir, "*.jpg")) + glob.glob(os.path.join(out_img_dir, "*.png"))
    object_crops = []

    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(out_lbl_dir, base_name + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img, _ = img.shape

        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = parts[0]
                cx, cy, w, h = map(float, parts[1:5])
                px, py = max(0, int((cx - w / 2) * w_img)), max(0, int((cy - h / 2) * h_img))
                pw, ph = min(w_img - px, int(w * w_img)), min(h_img - py, int(h * h_img))

                if pw > 5 and ph > 5:
                    crop = img[py:py+ph, px:px+pw]
                    object_crops.append((cls_id, crop))

    print(f"✅ 提取到 {len(object_crops)} 个候选特征切片！开始合成极小目标...")

    # 3. 随机合成粘贴
    created_cnt = 0
    while created_cnt < target_count:
        bg_path = random.choice(img_files)
        base_name = os.path.splitext(os.path.basename(bg_path))[0]
        lbl_path = os.path.join(out_lbl_dir, base_name + ".txt")

        img = cv2.imread(bg_path)
        if img is None: continue
        h_img, w_img, _ = img.shape

        cls_id, crop = random.choice(object_crops)
        ch, cw, _ = crop.shape

        # 强制按比例缩小到 15~28 像素 (标准小目标 AP_small < 1024 像素²)
        target_size = random.randint(15, 28)
        scale = target_size / max(ch, cw)
        new_w, new_h = max(4, int(cw * scale)), max(4, int(ch * scale))
        resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        paste_x = random.randint(10, w_img - new_w - 10)
        paste_y = random.randint(10, h_img - new_h - 10)

        # 图像融合并写回
        img[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized_crop
        cv2.imwrite(bg_path, img)

        # 计算 YOLO 归一化坐标并追加写回 txt
        cx, cy = (paste_x + new_w / 2) / w_img, (paste_y + new_h / 2) / h_img
        norm_w, norm_h = new_w / w_img, new_h / h_img

        with open(lbl_path, 'a') as f:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        created_cnt += 1

    print(f"🎉 扩增完成！共加入 {created_cnt} 个 AP_small 小目标！")
    print(f"📁 扩增后的数据集独立存储在:\n  - 图像: {out_img_dir}\n  - 标签: {out_lbl_dir}")

if __name__ == '__main__':
    safe_copy_paste_augmentation(target_count=180)