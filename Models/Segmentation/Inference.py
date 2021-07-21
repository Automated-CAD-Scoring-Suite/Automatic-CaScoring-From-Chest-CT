#
# Inference.py
# Author: Ahmad Abdalmageed
# Date: 6/10/21
#
import logging

import numpy as np
from skimage.transform import resize
import tensorflow as tf


class Infer:
    def __init__(self, model_path: str, model_input: tuple, threshold: float = 0.9, axial_first: bool = True):
        """
        Inference Model Script

        :param model_path: Tensor Flow's SavedModel Format Directory path
        :param model_input: The Saved Model's Input Shape, used in resizing Input Data
        :param threshold: Prediction Output Thresholding, useful in case of Binary Prediction
        :param axial_first: a Boolean Indicator that the Axial Slices appear first in the Input shape,
                            happens when loading Data in 3D Slicer.
        """
        self.path = model_path

        self.model_input = model_input
        self.data_shape = None
        self.threshold = threshold
        self.axial_first = axial_first

        self.model = tf.keras.models.load_model(self.path)

        # Define GPU Usage by the Server
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def predict(self, data: np.ndarray):
        slices = self.__prepare_data(data)
        res = self.model(slices)
        return self.__post_process(res)

    def __prepare_data(self, source: np.ndarray):
        # Copying and Saving Input Data Shape
        src = np.copy(source)

        # Move the Axial Axis to be the last Dimension to resemble Training Data
        if self.axial_first:
            src = np.moveaxis(src, 0, -1)
        self.data_shape = src.shape

        # Resizing Data to Fit Model.Input,
        src = resize(src, output_shape=self.model_input, preserve_range=True)

        # Convert input shape to TF Convention, This is a 3D model
        # Which needs about 5 Axis (Batches, L, W, D, Channels)
        src = np.expand_dims(src, 0)
        src = np.expand_dims(src, -1)

        # Scaling the Image to the Trained Data Scale, Data Is Ready to be Served
        src = self.range_scale(src)
        return src

    def __post_process(self, source):
        src = np.copy(source)

        # Fix Prediction Shape
        src = np.squeeze(src, 0)
        src = np.squeeze(src, -1)

        # Upscale to the Input Shape
        src = resize(src, output_shape=self.data_shape)

        # Fix Return Shape for Slicer
        if self.axial_first:
            src = np.moveaxis(src, -1, 0)

        # Thresholding output Prediction
        src[src < self.threshold] = 0.0
        src[src >= self.threshold] = 1.0

        return src

    @staticmethod
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


def ThresholdCAC(scan: np.ndarray, threshold: float = 130) -> np.ndarray:
    src = np.copy(scan)
    src[src < threshold] = 0
    src[src >= threshold] = 1
    return src


def QuantifyCAC(scan_threshold: np.ndarray, pred: np.ndarray, voxel_spacing: list) -> float:
    masked_out = scan_threshold * pred
    candidate_voxels = np.count_nonzero(masked_out)
    voxel_vol = (voxel_spacing[0]*voxel_spacing[1]*voxel_spacing[2]) / 3
    return candidate_voxels*voxel_vol


if __name__ == '__main__':
    import nibabel as nib

    heart_localization = Infer(model_path='./Models_Saved/Heart_Localization/',
                               model_input=(112, 112, 112),
                               axial_first=True)

    # load Scans
    Image = nib.load('./Data/Validation/ct_train_1019/imaging.nii.gz').get_fdata()
    print(Image.shape)
    Image = np.moveaxis(Image, -1, 0)

    # Predict using loaded model
    pred = heart_localization.predict(Image)

    print(pred.shape)
    print("Done Inference")
