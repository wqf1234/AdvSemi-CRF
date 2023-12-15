from skimage.metrics import structural_similarity as ssim
import cv2
import os

def calculate_ssim(image_folder_A, image_folder_B):
    ssim_values = []

    # 获取文件夹中的图像文件列表
    image_files_A = [f for f in os.listdir(image_folder_A) if os.path.isfile(os.path.join(image_folder_A, f))]
    image_files_B = [f for f in os.listdir(image_folder_B) if os.path.isfile(os.path.join(image_folder_B, f))]

    # 确保两个文件夹中图像数量相同
    assert len(image_files_A) == len(image_files_B), "两个文件夹中的图像数量不一致"

    # 遍历两个文件夹中的图像文件
    for file_A, file_B in zip(image_files_A, image_files_B):
        # 读取图像 A 和图像 B
        image_A = cv2.imread(os.path.join(image_folder_A, file_A), cv2.IMREAD_GRAYSCALE)
        image_B = cv2.imread(os.path.join(image_folder_B, file_B), cv2.IMREAD_GRAYSCALE)

        # 调整图像 B 到图像 A 的尺寸
        image_B_resized = cv2.resize(image_B, (image_A.shape[1], image_A.shape[0]))

        # 计算 SSIM
        ssim_value, _ = ssim(image_A, image_B_resized, full=True)
        ssim_values.append(ssim_value)

    # 计算均值
    mean_ssim = sum(ssim_values) / len(ssim_values)

    return mean_ssim

# 示例用法
folder_A = "C:/Users/24199/Desktop/QZT_label_transform"
folder_B = "C:/Users/24199/Desktop/QZTresults"

mean_ssim = calculate_ssim(folder_A, folder_B)
print(f"两个文件夹中图像的平均SSIM值: {mean_ssim}")
