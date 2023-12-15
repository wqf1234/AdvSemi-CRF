import os

def rename_files_by_index(folder_path, text_file_path):
    # 获取文件夹中的图像文件列表
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 按照文件名中的索引数字排序
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    # 读取文本文件中的文件名列表
    with open(text_file_path, 'r') as file:
        new_names = file.read().splitlines()

    # 遍历文件并重命名
    for i, file in enumerate(files):
        old_path = os.path.join(folder_path, file)
        new_name = new_names[i]
        new_path = os.path.join(folder_path, f"{new_name}.jpg")

        os.rename(old_path, new_path)
        print(f"Renamed: {file} to {new_name}.jpg")
# 用法示例
folder_path = "C:/Users/24199/Desktop/fsdownload/crop"
text_file_path = "C:/Users/24199/Desktop/fsdownload/val.txt"
rename_files_by_index(folder_path, text_file_path)
