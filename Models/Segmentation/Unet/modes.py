#
# modes.py
# Author: Ahmad Abdalmageed
# Date: 7/13/21
#
from tensorflow.keras.layers import Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, MaxPool2D, MaxPool3D,\
    AveragePooling2D, AveragePooling3D, UpSampling2D, UpSampling3D
from enum import Enum


# Modes Used by the User
class Transpose(Enum):
    transpose2D = Conv2DTranspose
    transpose3D = Conv3DTranspose


class UpSample(Enum):
    upSample2D = UpSampling2D
    upSample3D = UpSampling3D


class Conv(Enum):
    conv2D = Conv2D
    conv3D = Conv3D


class Pooling(Enum):
    max2D = MaxPool2D
    max3D = MaxPool3D
    average2D = AveragePooling2D
    average3D = AveragePooling3D
