import os
import shutil
import random

# ==================== 配置路径与比例 ====================
# 你的原始 YOLODataset 根目录
dataset_dir = r"E:\挑战sci\挑战sci\原始数据集\hrnavy-OHID-FF-76e995c卫星遥感1000+\YOLODataset"

# 新建的划分输出目录
output_dir = r"E:\挑战sci\挑战sci\datasetready\dataset_stage1_ohid"

# 划分比例 7:2:1
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 固定随机种子（确保每次运行划分结果完全一致，便于论文复现）
random.seed(42)
# =======================================================

images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

# 获取所有图片文件名（不含后缀）
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

# 打乱顺序
random.shuffle(image_files)

total_count = len(image_files)
train_count = int(total_count * train_ratio)
val_count = int(total_count * val_ratio)

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

print(f"数据总量: {total_count}")
print(f"训练集 (Train): {len(train_files)} 张 (70%)")
print(f"验证集 (Val): {len(val_files)} 张 (20%)")
print(f"测试集 (Test): {len(test_files)} 张 (10%)")

# 创建输出目录
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

def copy_files(file_list, split_name):
    for img_name in file_list:
        base_name = os.path.splitext(img_name)[0]
        
        # 复制图片
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(output_dir, 'images', split_name, img_name)
        shutil.copy(src_img, dst_img)
        
        # 复制对应的 label txt 文件（若为负样本没有 txt，会自动跳过或生成空 txt）
        src_label = os.path.join(labels_dir, base_name + '.txt')
        dst_label = os.path.join(output_dir, 'labels', split_name, base_name + '.txt')
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            # 纯背景负样本，生成空白 txt
            open(dst_label, 'w').close()

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("\n数据划分完成！新数据集已生成至:", output_dir)