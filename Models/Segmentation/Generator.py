#
# Implementation of a Custom Tensorflow Generator
#

import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom, rotate
from skimage.exposure import adjust_gamma
from skimage.transform import resize
import random


# Augmenter Class
class NiftyAugmentor:
    """Augmentor Class, Responsible for the application of different Augmentation Techniques"""

    def __init__(self, rotation: list = None, gamma: list = None, random_axis_flip: list = None):
        self.rotation_range = rotation
        self.gamma_range = gamma
        self.random_flip = random_axis_flip
        self.filters = [self.__rotate, self.__gamma, self.__flip]

    def fit(self, vector: np.ndarray) -> np.ndarray:
        """
        Main Filtering Functions that takes input Image and apply
        chosen filters.
        :param vector: Input n-Dimensional Image
        :return: Filtered n-Dimensional Image
        """
        src = np.copy(vector)
        for process in self.filters:
            if process is not None:
                src = process(src)
        return src

    def __rotate(self, volume: np.ndarray):
        """

        :param volume:
        :return:
        """
        if isinstance(self.rotation_range, list):
            angle = random.choice(self.rotation_range)
            src = np.copy(volume)
            return np.clip(rotate(src, angle, reshape=False), 0.0, 1.0)
        else:
            return None

    def __gamma(self, volume):
        if isinstance(self.gamma_range, list):
            gamma = random.choice(self.gamma_range)
            src = np.copy(volume)
            return adjust_gamma(src, gamma)
        else:
            return None

    def __flip(self, volume):
        if isinstance(self.random_flip, list):
            axis = random.choice(self.random_flip)
            src = np.copy(volume)
            return np.flip(src, axis)
        else:
            return None


# Keras Sequence Class
class NiftyGen(tf.keras.utils.Sequence):
    """Keras Sequence for loading Nifty image formats"""

    def __init__(self, images_path: str, mode: str = '2D', augmenter=None,
                 scale: bool = True, shuffle: bool = True, down_factor=None, output_shape=None, channels: int = 1, save: bool = False):
        self.path = images_path
        self.channels = channels
        self.down_factor = down_factor
        self.shuffle = shuffle
        self.scale = scale
        self.aug = augmenter
        self.mode = mode
        self.save_batch = save
        self.output_shape = output_shape
        self.modeIs2D = bool(mode == "2D")
        self.modeIs3D = bool(mode == "3D")
        self.target = "Batches"
        self.records = sorted(os.listdir(self.path))

    def __len__(self) -> int:
        return len(self.records)

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

    @staticmethod
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
        src_zoomed = np.clip(src_zoomed, 0.0, 1.0)
        return src_zoomed

    @staticmethod
    def ShuffleImg(img: np.ndarray) -> np.ndarray:
        """
            Shuffle Input image in the z direction
        :param img: Input Image nd array
        :return: Shuffled Image
        """
        np.random.seed(123)
        # Extract Indices of the z view
        idx = np.arange(img.shape[-1])
        np.random.shuffle(idx)
        return img[:, :, idx]

    def ProcessImage(self, img: np.ndarray, segmentation: bool = False) -> np.ndarray:
        """
            Process Input Image loaded from Local Training
            Folder, Apply Slicing, Dimension checks and Shuffling

        :param img:
        :param segmentation:
        :return:
        """
        src = np.copy(img)

        # Scale Image in
        if self.scale and not segmentation:
            src = self.range_scale(src)

        # Fix Scan Orientation
        src = rotate(src, 90)

        # Down Sample the image with the selected Factor
        if self.down_factor:
            # src = self.zoom3D(src, self.down_factor)
            src = resize(src, output_shape=self.output_shape)

        # Shuffle the Images in the Z direction
        if self.shuffle:
            src = self.ShuffleImg(src)
        return src

    def ReshapeImage(self, img: np.ndarray) -> np.ndarray:
        """
            Reshape the Image to be Compatible with TF Backend
        :param img: Input Image
        :return: Reshaped Image
        """
        src = np.copy(img)
        if self.aug:
            src = self.aug.fit(src)
        # Reshape the Output Images to be compatible with Tensorflow Slicing System
        # (batch_size, H, W, D, Channels)
        if self.modeIs3D:
            src = np.expand_dims(src, 0)
            src = np.expand_dims(src, -1)

        if self.modeIs2D:
            # H, W, S
            src = np.moveaxis(src, -1, 0)
            src = np.expand_dims(src, -1)
        return src

    def __getitem__(self, index: int) -> tuple:
        # Load Segmentation and Data in the Record
        image_path = os.path.join(self.path, self.records[index])
        img = nib.load(os.path.join(image_path, 'imaging.nii.gz')).get_fdata().astype('float32')
        seg = nib.load(os.path.join(image_path, 'segmentation.nii.gz')).get_fdata().astype('float32')

        # Both Image and Segmentation must be with the same dimensions
        assert (img.shape == seg.shape), \
            f'Images and Segmentation are with different Dimensions,{seg.shape} {img.shape}'

        # Process the Scan and Segmentation Arrays
        img = self.ProcessImage(img)
        seg = self.ProcessImage(seg, segmentation=True)
        s = img.shape[-1]
        # Concatenate Both Sources for a faster
        src = np.concatenate([img, seg], -1)
        # Reshape Both Arrays
        src = self.ReshapeImage(src)

        if self.modeIs3D:
            img = src[:, :, :, :s, :]
            seg = src[:, :, :, s:, :]
        if self.modeIs2D:
            img = src[:s, :, :, :]
            seg = src[s:, :, :, :]

        # Checking Segmentation only Contains 0 and 1, Fixing Rotations and Zooming Results
        seg[seg <= 0.5] = 0.0
        seg[seg > 0.5] = 1.0

        # Saving Created Batches at Batches Directory
        if self.save_batch:
            test_img = np.squeeze(img, -1)
            test_seg = np.squeeze(seg, -1)
            if self.modeIs3D:
                test_img = np.squeeze(img, 0)
                test_seg = np.squeeze(seg, 0)

            fig, ax = plt.subplots(1, 2, figsize=(20, 20))
            for IMAGE in range(0, test_img.shape[-1], 50):
                ax[0].imshow(test_img[IMAGE, :, :], "gray")
                ax[1].imshow(test_seg[IMAGE, :, :], "gray")

                if not os.path.isdir(self.target):
                    os.makedirs(self.target)
                plt.savefig(os.path.join(self.target, f"IMG_BATCH_{index}_{IMAGE}"))
        print(f"\nLoaded Sources with Shapes, {img.shape}, {seg.shape}", flush=True)
        return img, seg

    def on_epoch_end(self):
        """
        Randomly shuffle Images selected
        """
        random.shuffle(self.records)
