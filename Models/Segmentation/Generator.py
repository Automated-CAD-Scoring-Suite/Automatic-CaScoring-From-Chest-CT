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
import SimpleITK as sITK
import math


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
                 scale: bool = True, shuffle: bool = True, down_factor=None, output_shape=None,
                 channels: int = 1, save: bool = False):
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
            src = resize(src, output_shape=self.output_shape, preserve_range=True)

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

    def LoadData(self, index):
        """
        Data Loader Function
        :param index:
        :return:
        """

        image_path = os.path.join(self.path, self.records[index])
        img = nib.load(os.path.join(image_path, 'imaging.nii.gz')).get_fdata().astype('float32')
        seg = nib.load(os.path.join(image_path, 'seg_norm.nii.gz')).get_fdata().astype('float32')

        # Both Image and Segmentation must be with the same dimensions
        assert (img.shape == seg.shape), \
            f'Images and Segmentation are with different Dimensions,{seg.shape} {img.shape}'

        return img, seg

    def SaveBatch(self, img, seg, index):
        test_img = np.squeeze(img, -1)
        test_seg = np.squeeze(seg, -1)
        if self.modeIs3D:
            test_img = np.squeeze(img, 0)
            test_seg = np.squeeze(seg, 0)
            test_img = np.squeeze(test_img, -1)
            test_seg = np.squeeze(test_seg, -1)

        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        for IMAGE in range(0, test_img.shape[-1], 50):
            ax[0].imshow(test_img[IMAGE, :, :], "gray")
            ax[1].imshow(test_seg[IMAGE, :, :], "gray")

            if not os.path.isdir(self.target):
                os.makedirs(self.target)
            plt.savefig(os.path.join(self.target, f"IMG_BATCH_{index}_{IMAGE}"))

    def __getitem__(self, index: int) -> tuple:
        # Load Segmentation and Data in the Record
        img, seg = self.LoadData(index)

        # Process the Scan and Segmentation Arrays
        img = self.ProcessImage(img)
        seg = self.ProcessImage(seg, segmentation=True)

        # Concatenate Both Sources for a faster
        s = img.shape[-1]
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
            self.SaveBatch(img, seg, index)
        return img, seg

    def on_epoch_end(self):
        """
        Randomly shuffle Images selected
        """
        random.shuffle(self.records)


class CACGen(NiftyGen):
    """
     CAC Generator is a TF's Generator that Load Calcification Data
     and Generate Divided Cubes from Localized Heart

    """
    def LoadData(self, index):
        # Scans Paths
        image_path = os.path.join(self.path, self.records[index])

        # Load Patient's Scans
        sITK_image = sITK.ReadImage(os.path.join(image_path, 'imaging.mhd'))
        image_array = sITK.GetArrayFromImage(sITK_image)

        # Load Reference Standard Segmentation
        sITK_ref = sITK.ReadImage(os.path.join(image_path, 'seg_norm.mhd'))
        ref_array = sITK.GetArrayFromImage(sITK_ref)

    def getCubes(self, img, msk, cube_size):
        sizeX = img.shape[2]
        sizeY = img.shape[1]
        sizeZ = img.shape[0]

        cubeSizeX = cube_size[0]
        cubeSizeY = cube_size[1]
        cubeSizeZ = cube_size[2]

        n_z = int(math.ceil(float(sizeZ) / cubeSizeZ))
        n_y = int(math.ceil(float(sizeY) / cubeSizeY))
        n_x = int(math.ceil(float(sizeX) / cubeSizeX))

        sizeNew = [n_z * cubeSizeZ, n_y * cubeSizeY, n_x * cubeSizeX]

        imgNew = np.zeros(sizeNew, dtype=np.float16)
        imgNew[0:sizeZ, 0:sizeY, 0:sizeX] = img

        mskNew = np.zeros(sizeNew, dtype=np.int)
        mskNew[0:sizeZ, 0:sizeY, 0:sizeX] = msk

        n_ges = n_x * n_y * n_z
        n_4 = int(math.ceil(float(n_ges) / 4.) * 4)

        imgCubes = np.zeros((n_4, cubeSizeZ, cubeSizeY, cubeSizeX)) - 1  # -1 = air
        mskCubes = np.zeros((n_4, cubeSizeZ, cubeSizeY, cubeSizeX))

        count = 0
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    imgCubes[count] = imgNew[z * cubeSizeZ:(z + 1) * cubeSizeZ,
                                      y * cubeSizeY:(y + 1) * cubeSizeY,
                                      x * cubeSizeX:(x + 1) * cubeSizeX]
                    mskCubes[count] = mskNew[z * cubeSizeZ:(z + 1) * cubeSizeZ,
                                      y * cubeSizeY:(y + 1) * cubeSizeY,
                                      x * cubeSizeX:(x + 1) * cubeSizeX]
                    count += 1

        return imgCubes, mskCubes, n_x, n_y, n_z


if __name__ == '__main__':
    training = 'Data/Training/'
    mode = '3D'
    output_shape = (112, 112, 112)
    input_shape = (112, 112, 112, 1)
    down_factor = True

    aug = NiftyAugmentor([-10, 10, 0], [0.95, 1, 1.1], [0, 1])
    gen = NiftyGen(training, augmenter=aug, mode=mode, output_shape=output_shape, down_factor=down_factor, save=False,
                   shuffle=False)

    for i in gen:
        print(i[0].shape)
        print(i[1].shape)
