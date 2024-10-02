import os
import shutil
import random
def process_images(source_dir, target_dir, select_count, max_gap=None):
    folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]  # 获取所有子文件夹
    txt_file = os.path.join(os.path.dirname(source_dir), f"{os.path.basename(source_dir)}.txt")  # 假设txt文件名与源目录名相同
    if os.path.exists(txt_file):
        shutil.copy(txt_file, target_dir)  # 复制txt文件到目标文件夹
    for folder in folders:  # 处理实际的文件夹
        folder_path = os.path.join(source_dir, folder)
        if not os.path.exists(folder_path):
            continue
        
        images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png'))]
        
        # 根据选择的张数进行过滤
        if len(images) < 5 and select_count > len(images):
            continue
        
        # 随机选择指定数量的图像
        if select_count == 2 and max_gap is not None:
            selected_images = []
            indices = random.sample(range(len(images)), len(images))  # 随机打乱索引
            for i in range(len(indices)):
                if len(selected_images) < select_count:
                    if not selected_images or (indices[i] - indices[selected_images[-1]]) <= max_gap:
                        selected_images.append(indices[i])
            selected_images = [images[i] for i in selected_images]
        else:
            selected_images = random.sample(images, min(select_count, len(images)))
        
        
        # 创建目标文件夹
        target_folder = os.path.join(target_dir, folder)
        os.makedirs(target_folder, exist_ok=True)
        
        for img in selected_images:
            shutil.copy(os.path.join(folder_path, img), target_folder)

# 设置源目录和目标目录
train_source = '../train'  # 训练集路径
test_source = '../test'    # 测试集路径
train_target = '../trian_processed'  # 处理后的训练集路径
test_target = '../test_processed'    # 处理后的测试集路径

# 选择的张数
select_count = 2  # 可以是2, 3, 或4

# 处理训练集和测试集时添加间隔参数
process_images(train_source, train_target, select_count, max_gap=3)  # 训练集处理805个文件夹
process_images(test_source, test_target, select_count, max_gap=3)    # 测试集处理155个文件夹