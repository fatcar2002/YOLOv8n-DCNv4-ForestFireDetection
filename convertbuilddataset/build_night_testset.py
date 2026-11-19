import os
import shutil

# ==================== 1. 路径配置 ====================
# 你已经选好的夜间图片目录
src_night_img_dir = r"F:\ultralyticsyolo8\dataset_night_test\images\test"

# D-fire 原始测试集标签目录
src_dfire_lbl_dir = r"E:\挑战sci\挑战sci\原始数据集\D-fire\test\labels"

# 最终生成的夜间测试集根目录
dst_dataset_dir = r"F:\ultralyticsyolo8\dataset_night_test"
dst_img_dir = os.path.join(dst_dataset_dir, "images", "test")
dst_lbl_dir = os.path.join(dst_dataset_dir, "labels", "test")

os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

# ==================== 2. 扫描图片并对齐标注文件 ====================
print(f"正在扫描选好的夜间图片目录: {src_night_img_dir} ...")
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
img_files = [f for f in os.listdir(src_night_img_dir) if f.lower().endswith(valid_extensions)]

print(f"共扫描到夜间图片: {len(img_files)} 张")

matched_cnt = 0
missing_lbl_cnt = 0

for img_name in img_files:
    base_name = os.path.splitext(img_name)[0]
    lbl_name = f"{base_name}.txt"
    
    src_img_path = os.path.join(src_night_img_dir, img_name)
    src_lbl_path = os.path.join(src_dfire_lbl_dir, lbl_name)
    
    dst_img_path = os.path.join(dst_img_dir, img_name)
    dst_lbl_path = os.path.join(dst_lbl_dir, lbl_name)
    
    # 如果图片还不在 images/test 中，移动/复制过去
    if os.path.abspath(src_img_path) != os.path.abspath(dst_img_path):
        shutil.copy2(src_img_path, dst_img_path)
    
    # 匹配标注文件
    if os.path.exists(src_lbl_path):
        shutil.copy2(src_lbl_path, dst_lbl_path)
        matched_cnt += 1
    else:
        # 若原始标注不存在，创建空文件作为无目标样本处理
        with open(dst_lbl_path, 'w', encoding='utf-8') as f:
            pass
        missing_lbl_cnt += 1

print(f"\n🎉 夜间测试集构建完成！")
print(f"  ├─ 测试图片总数: {len(img_files)} 张 (已就位: {dst_img_dir})")
print(f"  ├─ 成功匹配真实标注: {matched_cnt} 个 (已保存: {dst_lbl_dir})")
if missing_lbl_cnt > 0:
    print(f"  └─ 补充生成空标注: {missing_lbl_cnt} 个")

# ==================== 3. 自动生成 dataset_night_test.yaml ====================
yaml_content = f"""# 选定夜间低照度测试集 (Night-time Robustness TestSet)
path: {dst_dataset_dir}
train: images/test
val: images/test
test: images/test

names:
  0: Smoke
  1: Fire
"""

yaml_path = os.path.join(dst_dataset_dir, "dataset_night_test.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"  └─ 测试集配置文件已生成: {yaml_path}")