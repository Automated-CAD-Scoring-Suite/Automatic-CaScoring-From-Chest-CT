#
# Implementation of a Custom Tensorflow Generator
#

import nibabel as nib
import numpy as np
import os
import tensorflow as tf
import random


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

    def __getitem__(self, index):
        # Load Segmentation and Data in the Record
        image_path = os.path.join(self.path, self.records[index])

        img = nib.load(os.path.join(image_path, 'imaging.nii.gz')).get_fdata()
        seg = nib.load(os.path.join(image_path, 'segmentation.nii.gz')).get_fdata()

        # Both Image and Segmentation must be with the same dimensions
        assert img.shape == seg.reshape, 'Images and Segmentation are with different Dimensions'

        # Scale in the HF Range
        if self.scale:
            img = self.range_scale(img)

        # Down sampling
        if self.down_factor:
            img = img[0::self.down_factor, 0::self.down_factor, :]
            seg = seg[0::self.down_factor, 0::self.down_factor, :]

        if self.shuffle:
            # Extract Indices of the z view
            idx = img.shape[-1]
            np.random.shuffle(idx)

            # Shuffle the z view
            img = img[:, :, idx]
            seg = seg[:, :, idx]

        # Slicing and Reshaping
        if self.channels == 1:
            img = img[:, :, self.batch_start: self.batch_start + self.batch_size]
        else:
            # Concatenate the same image self.channels times
            img = np.repeat(img[:, :, self.batch_start: self.batch_start + self.batch_size], self.channels, -1)

        seg = seg[:, :, self.batch_start: self.batch_start + self.batch_size]

        return (img.reshape((self.batch_size, img.shape[0], img.shape[1], self.channels)),
                seg.reshape((self.batch_size, img.shape[0], img.shape[1], 1)))

    def on_epoch_end(self):
        random.shuffle(self.records)
