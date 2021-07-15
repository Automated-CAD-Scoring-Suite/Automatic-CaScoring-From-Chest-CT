#
# Inference.py
# Author: Ahmad Abdalmageed
# Date: 6/10/21
#
import logging

import numpy as np
import torch
from scipy.ndimage import zoom
import SimpleITK as sitk


def load_itk(filename):
    """
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.

    :param filename:
    :return:
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


class Infer:
    def __init__(self, trace_path: str, model_path: str, axis: int = -1, slices: int = 20, shape: int = 352,
                 channels: str = 1):
        """
        Inference Model Script
        :param trace_path:
        :param axis:
        :param slices:
        :param shape
        :param channels:
        """
        self.trace = trace_path
        self.path = model_path
        self.axis = axis
        self.slices = slices
        self.shape = shape
        self.channels = channels
        self.model = torch.jit.load(self.trace)
        if torch.cuda.is_available():
            logging.info("GPU Processing")
            self.model.load_state_dict(torch.load(self.path), strict=False)
        else:
            logging.info("CPU Processing")
            self.model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu')), strict=False)

    def predict(self, data: np.ndarray):
        slices = self.__prepare_data(data)
        res = self.model(slices)
        return self.__post_process(res, threshold=0.5)

    def __prepare_data(self, source: np.ndarray):
        src = np.copy(source)
        src_shape = src.shape
        if len(src_shape) == 2:
            src = np.expand_dims(src, -1)
        src = self.range_scale(src)
        src = zoom(src, (self.shape / src_shape[0], self.shape / src_shape[0], 1))
        # print(src.shape)
        src = np.expand_dims(src, 0)
        if self.channels == 1:
            src = np.repeat(src, 3, 0)
        # print(src.shape)
        src = np.moveaxis(src, -1, 0)
        src = torch.Tensor(src)
        return src

    @staticmethod
    def __post_process(source, threshold: float):
        src = torch.nn.functional.upsample(source, size=(512, 512), mode='bilinear', align_corners=False)
        src = src.sigmoid().data.cpu().numpy().squeeze()
        src = (src - src.min()) / (src.max() - src.min() + 1e-8)
        src[src <= threshold] = 0.0
        src[src > threshold] = 1.0
        src = np.moveaxis(src, 0, -1)
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

# if __name__ == '__main__':
