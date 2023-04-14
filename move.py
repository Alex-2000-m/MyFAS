import os
import shutil

original_dir = r"D:\毕设\数据库论文\Oulu_Npu\test"

output_dir = r"D:\毕设\数据库论文\神经网络\毕设用的神经网络\test"

for root, dirs, files in os.walk(original_dir):
    if root.split("_")[-1] == "1":
        for file in files:
            original_file_path = os.path.join(root, file)
            a = os.path.basename(os.path.dirname(original_file_path))
            b = os.path.basename(original_file_path)
            output_file_path = os.path.normpath(os.path.join(output_dir + "/" + "1", a + "_" + b))  # 整理路径格式
            shutil.copyfile(original_file_path, output_file_path)

    elif root.split("_")[-1] == "2":
        for file in files:
            original_file_path = os.path.join(root, file)
            a = os.path.basename(os.path.dirname(original_file_path))
            b = os.path.basename(original_file_path)
            output_file_path = os.path.normpath(os.path.join(output_dir + "/" + "0", a + "_" + b))
            shutil.copyfile(original_file_path, output_file_path)
