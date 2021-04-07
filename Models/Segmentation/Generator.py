#
# Implementation of a Custom Tensorflow Generator
#
import nibabel as nib
import numpy as np
import os
import tensorflow as tf
import random
from scipy.ndimage import zoom, sobel
from skimage.exposure import equalize_hist, adjust_gamma
from functools import partial


# Augmenter Class
class NiftyAugmentor:
    """Augmentor Class, Responsible for the application of different Augmentation Techniques"""
    def __init__(self, equalize=True, adjust=True, sobel_transform=True, invert=True, gamma=2):
        self.eq = equalize_hist
        self.trans_laplace = partial(adjust_gamma, gamma=gamma)
        self.trans_sobel = sobel
        self.invert = lambda x: 1-x

        self.filters = {
            "equalize": [equalize, self.eq],
            "adjust": [adjust, self.trans_laplace],
            "sobel_transform": [sobel_transform, self.trans_sobel],
            "invert": [invert, self.invert],
        }
        self.vector = None
        self.filtered_result = []

    def fit(self, vector: np.ndarray):
        """
        Main Filtering Functions that takes input Image and apply
        chosen filters.
        :param vector: Input n-Dimensional Image
        :return: Filtered n-Dimensional Image
        """
        self.vector = np.copy(vector)

        for process in self.filters:
            if self.filters[process][0]:
                self.filtered_result.append(self.filters[process][1](self.vector))

        self.filtered_result = np.concatenate(self.filtered_result, -1)
        return np.concatenate([self.vector, self.filtered_result], -1)


# Keras Sequence Class
class NiftyGen(tf.keras.utils.Sequence):
    """Keras Sequence for loading Nifty image formats"""
    def __init__(self, images_path, batch_size, batch_start, augmenter: NiftyAugmentor = None, scale=True, shuffle=True,
                 down_factor=None, channels=1):
        self.path = images_path
        self.batch_size = batch_size
        self.batch_start = batch_start
        self.channels = channels
        self.down_factor = down_factor
        self.shuffle = shuffle
        self.scale = scale
        self.aug = augmenter
        self.records = sorted(os.listdir(self.path))

    def __len__(self):
        return len(self.records)

    @staticmethod
    def range_scale(img):
        """
        Scale Given Image Array using HF range
        :param img: Input 3d Array
        :return: Scaled Array
        """
        return (img - img.min()) / (img.max() - img.min())

    @staticmethod
    def zoom3D(img, factor: float):
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

        # For the Augmentation class input the image and concatenate the results
        if self.aug:
            img = self.aug.fit(img)

        # Reshape the Output Images to be compatible with Tensorflow Slicing System
        # (batch_size, Resolution, Resolution, Channels)
        return (img.reshape((self.batch_size, img.shape[0], img.shape[1], self.channels)),
                seg.reshape((self.batch_size, img.shape[0], img.shape[1], 1)))

    def on_epoch_end(self):
        """
        Randomly shuffle Images selected
        """
        random.shuffle(self.records)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img2 = nib.load('Data/Training/ct_train_1001/imaging.nii.gz').get_fdata()
    seg2 = nib.load('Data/Training/ct_train_1001/segmentation.nii.gz').get_fdata()

    scale = NiftyGen.range_scale
    img2 = scale(img2)

    aug = NiftyAugmentor()
    res = aug.fit(img2[:, :, 190:200])
    print(res.shape)

    fig, ax = plt.subplots(5, 10, figsize=(30, 30))

    s = 0
    for i in range(5):
        for j in range(10):
            ax[i][j].imshow(res[:, :, s], 'gray')
            s += 1
    plt.show()
