import os
import shutil
import glob

# ==================== 1. 路径配置 ====================
# 原始纯大雾图片路径
src_fog_dir = r"E:\挑战sci\挑战sci\原始数据集\野外森林火灾监GWFP\GWFP Images Classes\GWFP Images Classes\Fog\Fog Images"

# 生成的目标测试集根目录
dst_dataset_dir = r"F:\ultralyticsyolo8\dataset_fog_test"

# 采样间隔 (每 20 张抽 1 张)
sample_interval = 20

# ==================== 2. 创建目录结构 ====================
dst_img_dir = os.path.join(dst_dataset_dir, "images", "test")
dst_lbl_dir = os.path.join(dst_dataset_dir, "labels", "test")

os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

# 支持的图片格式
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG')

print(f"正在扫描原始文件夹: {src_fog_dir} ...")
all_images = [f for f in os.listdir(src_fog_dir) if f.endswith(valid_extensions)]
# 排序以保证按时间序列/文件名顺序均匀采样
all_images.sort()

total_count = len(all_images)
print(f"找到原始大雾图片共: {total_count} 张")

# ==================== 3. 均匀抽样与生成空标签 ====================
sampled_images = all_images[::sample_interval]
print(f"按照每 {sample_interval} 张抽 1 张的规则，共抽取: {len(sampled_images)} 张图片")

for idx, img_name in enumerate(sampled_images):
    # 1. 复制图片
    src_img_path = os.path.join(src_fog_dir, img_name)
    dst_img_path = os.path.join(dst_img_dir, img_name)
    shutil.copy2(src_img_path, dst_img_path)
    
    # 2. 生成对应的空txt文件 (负样本无目标框)
    base_name = os.path.splitext(img_name)[0]
    dst_lbl_path = os.path.join(dst_lbl_dir, f"{base_name}.txt")
    
    # 创建空文件
    with open(dst_lbl_path, 'w', encoding='utf-8') as f:
        pass  # 写入空内容

print(f"\n🎉 抽样完成！")
print(f"  └─ 测试图片已保存至: {dst_img_dir}")
print(f"  └─ 空txt标注已保存至: {dst_lbl_dir}")

# ==================== 4. 自动生成 dataset_fog_test.yaml ====================
yaml_content = f"""# 纯大雾抗虚警测试集 (Zero-Shot Negative TestSet)
path: {dst_dataset_dir}
train: images/test  # 借用位置填空
val: images/test
test: images/test

# 类别信息 (维持与训练集一致)
names:
  0: fire
  1: smoke
"""

yaml_path = os.path.join(dst_dataset_dir, "dataset_fog_test.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"  └─ 测试集配置文件已自动生成: {yaml_path}")