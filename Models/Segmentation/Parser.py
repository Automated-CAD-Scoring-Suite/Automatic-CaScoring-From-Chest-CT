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
nii_target_file_training = "Data4/Training"
nii_target_file_validate = "Data4/Validation"
target = None
target_image = None
target_mask = None

PARSE_Train = False
PARSE_Crop = False
out_crop = (384, 384, 80)
out_shape = (112, 112, 112)
SAVE_Nii = False


if PARSE_Train:
    target_image = train_target_image
    target_mask = train_target_mask
    target = training
else:
    target_image = valid_target_image
    target_mask = valid_target_mask
    target = validation


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
        IMAGING_NII = nib.load(IMAGING_PATH)
        SEGMENT_NII = nib.load(SEGMENT_PATH)

        IMAGING_DATA = IMAGING_NII.get_fdata()
        SEGMENT_DATA = SEGMENT_NII.get_fdata()

        print(f"Shapes {IMAGING_DATA.shape} {SEGMENT_DATA.shape}", flush=False)
        print(f"Preprocessing .. ", flush=False)

        IMAGING_DATA = range_scale(IMAGING_DATA)

        IMAGING_DATA = rotate(IMAGING_DATA, 90)
        SEGMENT_DATA = rotate(SEGMENT_DATA, 90)

        # Resizing the Volume
        # IMAGING_DATA = resize(IMAGING_DATA, out_shape)
        # SEGMENT_DATA = resize(SEGMENT_DATA, out_shape)

        # Thresholding the PreProcessed Segmentation
        SEGMENT_DATA[SEGMENT_DATA <= 0.5] = 0.0
        SEGMENT_DATA[SEGMENT_DATA > 0.5] = 1.0

        # Crop the Images to fit the Second Network
        if PARSE_Crop:
            # Get the Volume Midpoint
            mid_view = IMAGING_DATA.shape[0] // 2
            mid_depth = IMAGING_DATA.shape[-1] // 2
            mid_crop_view = out_crop[0] // 2
            mid_crop_depth = out_crop[-1] // 2

            IMAGING_DATA = IMAGING_DATA[mid_view - mid_crop_view: mid_view + mid_crop_view,
                                        mid_view - mid_crop_view: mid_view + mid_crop_view,
                                        mid_depth - mid_crop_depth: mid_depth + mid_crop_depth]

            SEGMENT_DATA = SEGMENT_DATA[mid_view - mid_crop_view: mid_view + mid_crop_view,
                                        mid_view - mid_crop_view: mid_view + mid_crop_view,
                                        mid_depth - mid_crop_depth: mid_depth + mid_crop_depth]

        # Create the Directory
        if not os.path.exists(target_image):
            os.makedirs(target_image)

        if not os.path.exists(target_mask):
            os.makedirs(target_mask)
        print(f"Slicing and Dicing .. ", flush=False)

        if SAVE_Nii:
            # Create Directories if they did not exist
            if not os.path.exists(nii_target_file_validate):
                os.makedirs(nii_target_file_validate)

            if not os.path.exists(nii_target_file_training):
                os.makedirs(nii_target_file_training)

            # Create a New NifTi Image
            IMAGING_NII = nib.Nifti1Image(IMAGING_DATA, IMAGING_NII.affine, IMAGING_NII.header)
            SEGMENT_NII = nib.Nifti1Image(SEGMENT_DATA, SEGMENT_NII.affine, SEGMENT_NII.header)
            f_path = None

            # Save the Processed Data as Nifty
            if PARSE_Train:
                f_path = nii_target_file_training
            else:
                f_path = nii_target_file_validate

            nib.save(IMAGING_NII, os.path.join(f_path, f'ct_train_{case}/imaging.nii.gz'))
            nib.save(SEGMENT_NII, os.path.join(f_path, f'ct_train_{case}/seg_norm.nii.gz'))

        else:
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
