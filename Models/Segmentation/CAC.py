#
# CAC.py
# Author: Ahmad Abdalmageed
# Date: 7/22/21
#
import numpy as np


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
