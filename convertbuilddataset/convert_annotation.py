import os
import shutil
import random

# ==================== 配置路径与比例 ====================
# 1. 原始解压好的 YOLODataset 绝对路径（请确保 labels 文件夹里确实有 txt 文件）
dataset_dir = r"E:\挑战sci\挑战sci\原始数据集\hrnavy-OHID-FF-76e995c卫星遥感1000+\YOLODataset"

# 2. 目标数据集目录
output_dir = r"F:\ultralyticsyolo8\dataset_stage1_ohid"

# 划分比例 7:2:1
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

random.seed(42)  # 保持固定的随机种子
# =======================================================

images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

# 兼容常见图片格式
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

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

# 清理并重新创建输出目录
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

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
        
        # 复制或补全对应的 label .txt 文件
        src_label = os.path.join(labels_dir, base_name + '.txt')
        dst_label = os.path.join(output_dir, 'labels', split_name, base_name + '.txt')
        
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            # 如果是纯背景负样本，生成空白 txt，防止 YOLO 找不到标签报错
            open(dst_label, 'w').close()

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("\n数据划分与补全完成！数据已更新至:", output_dir)