import os
import cv2
import numpy as np

def calculate_pixel_accuracy(gt_image, pred_image):
    # 将非零值设为1
    gt_mask = (gt_image > 0).astype(int)
    pred_mask = (pred_image > 0).astype(int)

    total_pixels = gt_mask.size
    correct_pixels = np.sum(gt_mask == pred_mask)
    accuracy = correct_pixels / total_pixels
    return accuracy

def calculate_iou(gt_image, pred_image):
    # 将非零值设为1
    gt_mask = (gt_image > 0).astype(int)
    pred_mask = (pred_image > 0).astype(int)

    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def evaluate_segmentation(gt_folder, pred_folder):
    gt_images = sorted(os.listdir(gt_folder))
    pred_images = sorted(os.listdir(pred_folder))

    pixel_accuracy_sum = 0
    iou_sum = 0

    for gt_name, pred_name in zip(gt_images, pred_images):
        gt_path = os.path.join(gt_folder, gt_name)
        pred_path = os.path.join(pred_folder, pred_name)

        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        pixel_accuracy = calculate_pixel_accuracy(gt_image, pred_image)
        iou = calculate_iou(gt_image, pred_image)

        pixel_accuracy_sum += pixel_accuracy
        iou_sum += iou

    num_images = len(gt_images)
    mean_pixel_accuracy = pixel_accuracy_sum / num_images
    mean_iou = iou_sum / num_images

    return mean_pixel_accuracy, mean_iou

# 示例用法
gt_folder_path = "C:/Users/24199/Desktop/DZTlabel"
pred_folder_path = "C:/Users/24199/Desktop/DZTcrf"

pixel_accuracy, iou = evaluate_segmentation(gt_folder_path, pred_folder_path)

print(f"Mean Pixel Accuracy: {pixel_accuracy}")
print(f"Mean IoU: {iou}")
