#
# Parser.py
# Author: Ahmad Abdalmageed
# Date: 5/21/21
#
import nibabel as nib
from scipy.ndimage import zoom, rotate
from skimage.transform import resize
import numpy as np
import cv2
import os

training = 'Data/Training/'
validation = 'Data/Validation/'
train_target_image = 'Data3/Training/image'
train_target_mask = 'Data3/Training/mask'
valid_target_image = 'Data3/Validation/image'
valid_target_mask = 'Data3/Validation/mask'
target = None
target_image = None
target_mask = None

PARSE_Train = False
PARSE_Crop = False
out_shape = (112, 112, 112)

if PARSE_Train:
    target_image = train_target_image
    target_mask = train_target_mask
    target = training
else:
    target_image = valid_target_image
    target_mask = valid_target_mask
    target = validation


def find_roi_2D(image_slice):
    # rotate -90
    s_rotated = np.rot90(image_slice, k=3)

    # flip slice
    s_fliped = np.flip(image_slice, axis=0)
    s_rotated_fliped = np.flip(s_rotated, axis=0)

    # Get up and down coordinates
    y1 = np.unravel_index(np.argmax(image_slice, axis=None), image_slice.shape)
    y2 = np.unravel_index(np.argmax(s_fliped, axis=None), image_slice.shape)

    x1 = np.unravel_index(np.argmax(s_rotated, axis=None), image_slice.shape)
    x2 = np.unravel_index(np.argmax(s_rotated_fliped, axis=None), image_slice.shape)

    # return x1, x2, y1, y2 of image
    return x1[0], image_slice.shape[1] - x2[0], y1[0], image_slice.shape[0] - y2[0]


def get_coords(Slices: list):
    """
    Returns shape[1] then shape[0]
    :param Slices: list of 3 slices
    :return: list [x1,x2,y1,y2]
    """

    # Initialize coordinates
    x1 = list()
    x2 = list()
    y1 = list()
    y2 = list()

    # Find ROI in each slice
    for _slice in Slices:
        pnt1, pnt2, pnt3, pnt4 = find_roi_2D(_slice)
        x1.append(pnt1)
        x2.append(pnt2)
        y1.append(pnt3)
        y2.append(pnt4)

    # Return shape[1] then shape[0]
    return [min(x1), max(x2), min(y1), max(y2)]


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


if __name__ == '__main__':
    FILES = [file for file in sorted(os.listdir(target)) if file.startswith("ct_train")]

    for FILE in FILES:
        case = FILE[-4:]
        # Get each case File Path
        FILE_PATH = os.path.join(target, FILE)

        # Extract the Files Paths
        IMAGING_FILES = [image for image in sorted(os.listdir(FILE_PATH)) if image.endswith("imaging.nii.gz")][0]
        SEGMENT_FILES = [segment for segment in sorted(os.listdir(FILE_PATH))
                         if segment.endswith("seg_norm.nii.gz")][0]

        IMAGING_PATH = os.path.join(FILE_PATH, IMAGING_FILES)
        SEGMENT_PATH = os.path.join(FILE_PATH, SEGMENT_FILES)
        print(f"Loading Case {IMAGING_PATH} .. ", flush=False)

        # Load the File
        IMAGING_DATA = nib.load(IMAGING_PATH).get_fdata()
        SEGMENT_DATA = nib.load(SEGMENT_PATH).get_fdata()

        print(f"Shapes {IMAGING_DATA.shape} {SEGMENT_DATA.shape}", flush=False)
        print(f"Preprocessing .. ", flush=False)

        IMAGING_DATA = range_scale(IMAGING_DATA)

        IMAGING_DATA = rotate(IMAGING_DATA, 90)
        SEGMENT_DATA = rotate(SEGMENT_DATA, 90)

        # Resizing the Volume
        # IMAGING_DATA = zoom3D(IMAGING_DATA, 2)
        # SEGMENT_DATA = zoom3D(SEGMENT_DATA, 2)
        # IMAGING_DATA = resize(IMAGING_DATA, out_shape)
        # SEGMENT_DATA = resize(SEGMENT_DATA, out_shape)

        # Thresholding the PreProcessed Segmentation
        SEGMENT_DATA[SEGMENT_DATA <= 0.5] = 0.0
        SEGMENT_DATA[SEGMENT_DATA > 0.5] = 1.0

        # Create the Directory
        if not os.path.exists(target_image):
            os.makedirs(target_image)

        if not os.path.exists(target_mask):
            os.makedirs(target_mask)

        print(f"Slicing and Dicing .. ", flush=False)
        for s in range(IMAGING_DATA.shape[-1]):
            # Slicing the Required Axis
            z_img = IMAGING_DATA[:, :, s]
            z_seg = SEGMENT_DATA[:, :, s]

            # Saving as PNG
            cv2.imwrite(os.path.join(target_image, f"{case}_{s:04}_0.png"), z_img * 255)
            cv2.imwrite(os.path.join(target_mask, f"{case}_{s:04}_0.png"), z_seg * 255)

        for s in range(IMAGING_DATA.shape[1]):
            # Slicing the Required Axis
            z_img = IMAGING_DATA[:, s, :]
            z_seg = SEGMENT_DATA[:, s, :]

            # Saving as PNG
            cv2.imwrite(os.path.join(target_image, f"{case}_{s:04}_1.png"), z_img * 255)
            cv2.imwrite(os.path.join(target_mask, f"{case}_{s:04}_1.png"), z_seg * 255)

        for s in range(IMAGING_DATA.shape[0]):
            # Slicing the Required Axis
            z_img = IMAGING_DATA[s, :, :]
            z_seg = SEGMENT_DATA[s, :, :]

            # Saving as PNG
            cv2.imwrite(os.path.join(target_image, f"{case}_{s:04}_2.png"), z_img * 255)
            cv2.imwrite(os.path.join(target_mask, f"{case}_{s:04}_2.png"), z_seg * 255)
        print("Saved !!", flush=False)
