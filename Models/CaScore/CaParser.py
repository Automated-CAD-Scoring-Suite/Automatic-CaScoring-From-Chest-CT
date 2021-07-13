#
# CaParser.py
# Author: Ahmad Abdalmageed
# Date: 5/20/21
#
import os
import SimpleITK as sITK
import numpy as np
import cv2
from scipy.ndimage import zoom


def zoom3D(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Down Sample the input volume to desired shape
    :param img: Input Img
    :param factor: Down sampling Factor
    :return: Zoomed Img
    """
    src = np.copy(img)
    z_factor = (src.shape[-1] * factor) / src.shape[0]
    src_zoomed = zoom(src, (1 / factor, 1 / factor, 1 / z_factor))
    return src_zoomed


def range_scale(img) -> np.ndarray:
    """
        Scale Given Image Array using HF range
    :param img: Input 3d Array
    :return: Scaled Array
    """
    src = np.copy(img)
    min_bound = -1024.0
    max_bound = 1354.0
    src = (src - min_bound) / (max_bound - min_bound)
    src[src > 1] = 1.
    src[src < 0] = 0.
    return src


image_path = './Data/Training/Images'
ref_path = './Data/Training/Reference standard'
target_dir = './CaDataPNG'
target_dir_ref = './CaDataPNG/masks'
target_dir_images = './CaDataPNG/images'
Parse_Train = True

training_cti_images = sorted([image for image in os.listdir(image_path) if image.endswith('cti.mhd')])
training_reference = sorted([image for image in os.listdir(ref_path) if image.endswith('r.mhd')])
print(len(training_cti_images))
print(len(training_reference))

# Split the Data for Training and Validation
if Parse_Train:
    training_cti_images = training_cti_images[:-5]
    training_reference = training_reference[:-5]
else:
    training_cti_images = training_cti_images[-5:]
    training_reference = training_reference[-5:]
print(training_cti_images)
print(training_reference)

# Count the total volume of Calcifications for the Class Balancing problem
class_count = 0
for scans in zip(training_cti_images, training_reference):
    # Load Patient's Scans
    sITK_image = sITK.ReadImage(os.path.join(image_path, scans[0]))
    image_array = sITK.GetArrayFromImage(sITK_image)

    # Load Reference Standard Segmentation
    sITK_ref = sITK.ReadImage(os.path.join(ref_path, scans[1]))
    ref_array = sITK.GetArrayFromImage(sITK_ref)

    # Check Dimensions
    assert ref_array.shape == image_array.shape, "Shape Mis-match"

    # Scale the Scan's Values
    image_array = range_scale(image_array)

    # Convert Multi-class Reference to Binary Class
    ref_array[ref_array > 1] = 1

    # Check max and min values of the Reference Segmentation
    print(ref_array.min(), ref_array.max())
    ref_class_count = np.unique(ref_array, return_counts=True)
    print(ref_class_count)

    # Create Saving Directories if does not exist
    if not os.path.exists(target_dir):
        print("Directory Created !!")
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, 'images'))
        os.makedirs(os.path.join(target_dir, 'masks'))

    # Saving each Slice in both volumes into png images
    print("Slicing Volume .. ")
    for _slice in range(image_array.shape[0]):
        # Images
        cv2.imwrite(f"{os.path.join(target_dir_images, scans[0])}_{_slice}.png", image_array[_slice, :, :] * 255)

        # Scans
        cv2.imwrite(f"{os.path.join(target_dir_ref, scans[1])}_{_slice}.png", ref_array[_slice, :, :] * 255)
    print("Volume Saved Successfully ")
