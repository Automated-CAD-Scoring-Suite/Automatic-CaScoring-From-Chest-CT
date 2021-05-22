#
# Parser.py
# Author: Ahmad Abdalmageed
# Date: 5/21/21
#
import nibabel as nib
from scipy.ndimage import zoom, rotate
import numpy as np
import cv2
import os

training = 'Data/Training/'
target_image = 'Data2/image'
target_mask = 'Data2/mask'
target_npy_train = 'Data3/train_npz'
target_npy_test = 'Data3/test_vol_h5'


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
    SAVE_NPY = True
    FILES = [file for file in sorted(os.listdir(training)) if file.startswith("ct_train")]

    for FILE in FILES:
        case = FILE[-2:]
        # Get each case File Path
        FILE_PATH = os.path.join(training, FILE)

        # Extract the Files Paths
        IMAGING_FILES = [image for image in sorted(os.listdir(FILE_PATH)) if image.endswith("imaging.nii.gz")][0]
        SEGMENT_FILES = [segment for segment in sorted(os.listdir(FILE_PATH))
                         if segment.endswith("segmentation.nii.gz")][0]

        IMAGING_PATH = os.path.join(FILE_PATH, IMAGING_FILES)
        SEGMENT_PATH = os.path.join(FILE_PATH, SEGMENT_FILES)
        print(f"Loading Case {IMAGING_PATH} .. ", flush=False)

        # Load the File
        IMAGING_DATA = nib.load(IMAGING_PATH).get_fdata()
        SEGMENT_DATA = nib.load(SEGMENT_PATH).get_fdata()

        print(f"Shapes {IMAGING_DATA.shape} {SEGMENT_DATA.shape}", flush=False)
        print(f"Preprocessing .. ", flush=False)

        IMAGING_DATA = rotate(IMAGING_DATA, 90)
        SEGMENT_DATA = rotate(SEGMENT_DATA, 90)

        IMAGING_DATA = range_scale(IMAGING_DATA)

        IMAGING_DATA = zoom3D(IMAGING_DATA, 2)
        SEGMENT_DATA = zoom3D(SEGMENT_DATA, 2)

        SEGMENT_DATA[SEGMENT_DATA <= 0.5] = 0.0
        SEGMENT_DATA[SEGMENT_DATA > 0.5] = 1.0

        # Save Vol for TransUnet
        if SAVE_NPY:
            if not os.path.exists(target_npy_test):
                os.makedirs(target_npy_test)
            np.savez(os.path.join(target_npy_test, f"case{case}.npy.h5"),
                     image=IMAGING_DATA,
                     label=SEGMENT_DATA)

        # Create the Directory
        if not os.path.exists(target_image):
            os.makedirs(target_image)

        if not os.path.exists(target_mask):
            os.makedirs(target_mask)

        print(f"Slicing and Dicing .. ", flush=False)
        for s in range(IMAGING_DATA.shape[0]):
            # Slicing the Required Axis
            z_img = IMAGING_DATA[:, :, s]
            z_seg = SEGMENT_DATA[:, :, s]

            # Saving as PNG
            cv2.imwrite(os.path.join(target_image, f"{case}0{s}.png"), z_img * 255)
            cv2.imwrite(os.path.join(target_mask, f"{case}0{s}.png"), z_seg * 255)

            # Saving as NPZ
            if SAVE_NPY:
                if not os.path.exists(target_npy_train):
                    os.makedirs(target_npy_train)
                np.savez(os.path.join(target_npy_train, f"case{case}_slice{s}.npz"),
                         image=z_img,
                         label=z_seg)
        print("Saved !!", flush=False)

