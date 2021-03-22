#
# Implementation of a Custom Tensorflow Generator
#

import nibabel as nib
import numpy as np
import os
import tensorflow as tf
import random
from scipy.ndimage import zoom


# Keras Sequence Class
class NiftyGen(tf.keras.utils.Sequence):
    """
    Keras Sequence for loading Nifty image formats
    """
    def __init__(self, images_path, batch_size, batch_start, scale=True, shuffle=True, down_factor=None, channels=1):
        self.path = images_path
        self.batch_size = batch_size
        self.batch_start = batch_start
        self.channels = channels
        self.down_factor = down_factor
        self.shuffle = shuffle
        self.scale = scale
        self.records = sorted(os.listdir(self.path))

    def __len__(self):
        return len(self.records)

    def range_scale(self, img):
        """
        Scale Given Image Array using HF range
        :param img: Input 3d Array
        :return: Scaled Array
        """
        return (img - img.min()) / (img.max() - img.min())

    def zoom3D(self, img, factor: float):
        """
        Down Sample the input volume to desired shape
        :param img: Input Img
        :param factor: Down sampling Factor
        :return: Zoomed Img
        """
        assert img.shape[0] == img.shape[1], f"First View is not a Square, {img.shape}"
        z_factor = (img.shape[0]*factor)/img.shape[-1]
        return zoom(img, (factor, factor, z_factor))

    def __getitem__(self, index):
        # Load Segmentation and Data in the Record
        image_path = os.path.join(self.path, self.records[index])

        img = nib.load(os.path.join(image_path, 'imaging.nii.gz')).get_fdata()
        seg = nib.load(os.path.join(image_path, 'segmentation.nii.gz')).get_fdata()

        # Both Image and Segmentation must be with the same dimensions
        assert (img.shape == seg.shape),\
            f'Images and Segmentation are with different Dimensions,{seg.shape} {img.shape}'

        # Scale in the HF Range
        if self.scale:
            img = self.range_scale(img)

        # Down sampling
        if self.down_factor:
            img = self.zoom3D(img, self.down_factor)
            seg = self.zoom3D(seg, self.down_factor)

        # Shuffle Inputs
        if self.shuffle:
            # Extract Indices of the z view
            idx = np.arange(img.shape[-1])
            np.random.shuffle(idx)

            # Shuffle the z view
            img = img[:, :, idx]
            seg = seg[:, :, idx]

        # TODO: CONVERT to function, Enhance Readability
        # Taking Frames in the size of (batch_start - batch_size)
        # Fill RGB Channels with the Same Slice

        if self.channels == 1:
            img = img[:, :, self.batch_start: self.batch_start + self.batch_size]
        else:
            img = np.repeat(img[:, :, self.batch_start: self.batch_start + self.batch_size], self.channels, -1)

        seg = seg[:, :, self.batch_start: self.batch_start + self.batch_size]

        # Reshape the Output Images to be compatible with Tensorflow Slicing System
        # (batch_size, Resolution, Resolution, Channels)
        return (img.reshape((self.batch_size, img.shape[0], img.shape[1], self.channels)),
                seg.reshape((self.batch_size, img.shape[0], img.shape[1], 1)))

    def on_epoch_end(self):
        """
        Randomly shuffle Images selected
        """
        random.shuffle(self.records)


class NiftyAugmentor:
    pass
