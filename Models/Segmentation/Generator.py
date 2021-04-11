#
# Implementation of a Custom Tensorflow Generator
#
import nibabel as nib
import numpy as np
import os
import tensorflow as tf
import random
from scipy.ndimage import sobel
from skimage.exposure import equalize_hist, adjust_gamma
from functools import partial


# Augmenter Class
class NiftyAugmentor:
    """Augmentor Class, Responsible for the application of different Augmentation Techniques"""

    def __init__(self, equalize: bool = True, adjust: bool = True, sobel_transform: bool = True,
                 invert: bool = True, gamma: int = 2):
        self.eq = equalize_hist
        self.trans_laplace = partial(adjust_gamma, gamma=gamma)
        self.trans_sobel = sobel
        self.invert = lambda x: 1 - x
        self.filtered_result = None

        self.filters = {
            "equalize": [equalize, self.eq],
            "adjust": [adjust, self.trans_laplace],
            "sobel_transform": [sobel_transform, self.trans_sobel],
            "invert": [invert, self.invert],
        }

    def fit(self, vector: np.ndarray) -> np.ndarray:
        """
        Main Filtering Functions that takes input Image and apply
        chosen filters.
        :param vector: Input n-Dimensional Image
        :return: Filtered n-Dimensional Image
        """
        self.filtered_result = []

        for process in self.filters:
            if self.filters[process][0]:
                self.filtered_result.append(self.filters[process][1](vector))

        self.filtered_result = np.concatenate(self.filtered_result, -1)
        return np.concatenate([vector, self.filtered_result], -1)


# Keras Sequence Class
class NiftyGen(tf.keras.utils.Sequence):
    """Keras Sequence for loading Nifty image formats"""

    def __init__(self, images_path: str, batch_size: int, batch_start: int, augmenter: NiftyAugmentor = None,
                 scale: bool = True, shuffle: bool = True, down_factor=None, channels: int = 1):
        self.path = images_path
        self.batch_size = batch_size
        self.batch_start = batch_start
        self.channels = channels
        self.down_factor = down_factor
        self.shuffle = shuffle
        self.scale = scale
        self.aug = augmenter
        self.records = sorted(os.listdir(self.path))
        self.filter_size = 1
        if self.aug:
            # Calculate the number of filters added in
            # Augmentation process and +1 for the
            # Concatenation of the original image
            self.filter_size += len(self.aug.filters)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def range_scale(img) -> np.ndarray:
        """
        Scale Given Image Array using HF range
        :param img: Input 3d Array
        :return: Scaled Array
        """
        return (img - img.min()) / (img.max() - img.min())

    @staticmethod
    def zoom3D(img: np.ndarray, factor: float) -> np.ndarray:
        """
        Down Sample the input volume to desired shape
        :param img: Input Img
        :param factor: Down sampling Factor
        :return: Zoomed Img
        """
        return img[::factor, ::factor, :]

    @staticmethod
    def ShuffleImg(img: np.ndarray) -> np.ndarray:
        """
            Shuffle Input image in the z direction
        :param img: Input Image nd array
        :return: Shuffled Image
        """
        # Extract Indices of the z view
        idx = np.arange(img.shape[-1])
        np.random.shuffle(idx)

        return img[:, :, idx]

    def SliceImg(self, img: np.ndarray) -> np.ndarray:
        """
            Slice Given image with the batch size
        :param img: Input Image
        :return: Slice Array
        """
        # Check if Input Image is RGB or 1 Channel
        if self.channels == 1:
            return img[:, :, self.batch_start: self.batch_start + self.batch_size]
        else:
            # Taking Frames in the size of (batch_start - batch_size)
            # Fill RGB Channels with the Same Slice
            return np.repeat(img[:, :, self.batch_start: self.batch_start + self.batch_size], self.channels, -1)

    def ProcessImage(self, img: np.ndarray) -> np.ndarray:
        """
            Process Input Image loaded from Local Training
            Folder, Apply Slicing, Dimension checks and Shuffling
        :return: Processed Image
        """
        # Scale Image in range of 0 - 1
        if self.scale:
            img = self.range_scale(img)
        # Down Sample the image with the selected Factor
        if self.down_factor:
            img = self.zoom3D(img, self.down_factor)
        # Shuffle the Images in the Z direction
        if self.shuffle:
            img = self.ShuffleImg(img)
        return img

    def ReshapeImage(self, img: np.ndarray, channels: int, segmentation: bool) -> np.ndarray:
        """
            Reshape the Image to be Compatible with TF Backend
        :param img: Input Image
        :param channels: Channels of the Reshaped Image
        :param segmentation: Boolean indicates that the input image
                             is a segmentation image
        :return: Reshaped Image
        """
        if self.aug:
            if segmentation:
                # In case of Segmentation image input
                # Repeat the input segmentation to match
                # Augmented CT Scan Image
                img = np.repeat(img, self.filter_size, -1)
            else:
                # Apply the Augmentation Techniques in case of
                # Input Original Image
                img = self.aug.fit(img)
                img = img.astype("float32")

        # Reshape the Output Images to be compatible with Tensorflow Slicing System
        # (batch_size, Resolution, Resolution, Channels)
        return img.reshape((self.batch_size * self.filter_size, img.shape[0], img.shape[1], channels))

    def __getitem__(self, index: int) -> tuple:
        # Load Segmentation and Data in the Record
        image_path = os.path.join(self.path, self.records[index])
        img = nib.load(os.path.join(image_path, 'imaging.nii.gz')).get_fdata()
        seg = nib.load(os.path.join(image_path, 'segmentation.nii.gz')).get_fdata()

        # Both Image and Segmentation must be with the same dimensions
        assert (img.shape == seg.shape), \
            f'Images and Segmentation are with different Dimensions,{seg.shape} {img.shape}'

        # Process the Scan and Segmentation Arrays
        img = self.ProcessImage(img)
        seg = self.ProcessImage(seg)

        # Slice the Image with the given batch size
        img = self.SliceImg(img)
        seg = self.SliceImg(seg)

        # Reshape Both Arrays
        seg = self.ReshapeImage(seg, channels=1, segmentation=True)
        img = self.ReshapeImage(img, channels=1, segmentation=False)

        return img, seg

    def on_epoch_end(self):
        """
        Randomly shuffle Images selected
        """
        random.shuffle(self.records)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # img2 = nib.load('Data/Training/ct_train_1001/imaging.nii.gz').get_fdata()
    # seg2 = nib.load('Data/Training/ct_train_1001/segmentation.nii.gz').get_fdata()

    # scale = NiftyGen.range_scale
    # img2 = scale(img2)
    # seg2 = seg2[:, :, 190: 195]

    # aug = NiftyAugmentor()
    # res = aug.fit(img2[:, :, 190:195])
    # print(res.shape)

    # fig, ax = plt.subplots(6, 5, figsize=(30, 30))

    # s = 0
    # for i in range(5):
    #     for j in range(5):
    #         ax[i][j].imshow(res[:, :, s], 'gray')
    #         s += 1

    # for i in range(5):
    #     ax[5][i].imshow(seg2[:, :, i], 'gray')
    # plt.show()

    # seg3 = np.repeat(seg2, 5, -1)
    # print(seg3.shape)
    # print(res.shape)
    gen = NiftyGen('./Data/Training', 20, 0, NiftyAugmentor(), down_factor=4)

    for i in range(len(gen)):
        print("img", gen[i][0].shape)
        print("seg", gen[i][1].shape)
