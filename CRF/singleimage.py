import os

import numpy as np
import cv2
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels

def apply_crf(image, predictions):
    """
    Apply Conditional Random Field (CRF) as post-processing to segmentation results.

    Parameters:
    - image: Input image (numpy array).
    - predictions: Segmentation predictions (numpy array).

    Returns:
    - Post-processed segmentation results.
    """
    # predictions_gray = cv2.cvtColor(predictions, cv2.COLOR_RGB2GRAY)
    predictions_gray = predictions[:, :, 0]
    predictions_adjusted = np.where(predictions_gray == 128, 1, predictions_gray)
    print(np.unique(predictions_adjusted))

    # Convert predictions to unary potentials
    unary = unary_from_labels(predictions_adjusted, n_labels=2, gt_prob=0.7, zero_unsure=False)

    # Create a CRF model
    d = densecrf.DenseCRF2D(image.shape[1], image.shape[0], 2)

    # Add unary potentials to the CRF model
    d.setUnaryEnergy(unary)

    # Add pairwise potentials (sxy and srgb are CRF parameters)
    d.addPairwiseGaussian(sxy=(10,10), compat=5)
    d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=image, compat=5)


    # Run inference to obtain refined predictions
    refined_predictions = np.argmax(d.inference(3), axis=0).reshape(image.shape[:2])

    return refined_predictions

# Load your image and segmentation result here
image = cv2.imread("C:/Users/24199/Desktop/fsdownload/crop/cropped_image_0.jpg")  # Replace with the actual image path
predictions = cv2.imread("C:/Users/24199/Desktop/DZTresults/DZT_dj_030.png")  # Replace with the actual predictions file path
image_resized = cv2.resize(image, (predictions.shape[1], predictions.shape[0]))


print(np.unique(predictions))

# Apply CRF post-processing
refined_predictions = apply_crf(image_resized, predictions)

# Convert to RGB using a color mapping
label_color_mapping = {
    0: [0, 0, 0],      # Background (black)
    1: [128, 0, 0]   # Object (gray)
}

refined_predictions = cv2.resize(refined_predictions, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
refined_predictions_rgb = np.zeros_like(image)
for label, color in label_color_mapping.items():
    refined_predictions_rgb[refined_predictions == label] = color

# print(np.unique(refined_predictions_rgb))

cv2.imshow("Refined Predictions", refined_predictions_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()