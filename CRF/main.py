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

    # Convert predictions to unary potentials
    unary = unary_from_labels(predictions_adjusted, n_labels=2, gt_prob=0.7, zero_unsure=False)

    # Create a CRF model
    d = densecrf.DenseCRF2D(image.shape[1], image.shape[0], 2)

    # Add unary potentials to the CRF model
    d.setUnaryEnergy(unary)

    # Add pairwise potentials (sxy and srgb are CRF parameters)
    d.addPairwiseGaussian(sxy=(5,5), compat=5)
    d.addPairwiseBilateral(sxy=(5, 5), srgb=(13, 13, 13), rgbim=image, compat=5)


    # Run inference to obtain refined predictions
    refined_predictions = np.argmax(d.inference(5), axis=0).reshape(image.shape[:2])

    return refined_predictions

# Load your image and segmentation result here
input_floder = "C:/Users/24199/Desktop/QZTimages/"  # Replace with the actual image path
predictions_floder = "C:/Users/24199/Desktop/QZTresults/"  # Replace with the actual predictions file path
output_floder = "C:/Users/24199/Desktop/QZTcrf/"

# print(np.unique(predictions))


# Iterate through all images in the input folder
for filename in os.listdir(input_floder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Load image
        image_path = os.path.join(input_floder, filename)
        image = cv2.imread(image_path)

        # Load segmentation predictions
        predictions_filename = os.path.splitext(filename)[0] + ".png"
        predictions_path = os.path.join(predictions_floder, predictions_filename)
        predictions = cv2.imread(predictions_path)

        image_resized = cv2.resize(image, (predictions.shape[1], predictions.shape[0]))

        # Apply CRF
        refined_predictions = apply_crf(image_resized, predictions)

        # Convert to RGB using a color mapping
        label_color_mapping = {
            0: [0, 0, 0],  # Background (black)
            1: [128, 0, 0]  # Object (gray)
        }

        refined_predictions = cv2.resize(refined_predictions, (image.shape[1], image.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
        refined_predictions_rgb = np.zeros_like(image)
        for label, color in label_color_mapping.items():
            refined_predictions_rgb[refined_predictions == label] = color

        # Save the refined predictions
        output_path = os.path.join(output_floder, filename)
        cv2.imwrite(output_path, refined_predictions_rgb)

print("CRF post-processing completed.")