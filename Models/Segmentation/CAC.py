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


def QuantifyCAC(scan_threshold: np.ndarray, pred: np.ndarray, voxel_spacing: list):
    masked_out_pred = scan_threshold * pred
    shape = scan_threshold.shape
    # Cropped = np.copy(scan_threshold)
    # Cropped[0:4, 0:40, :] = 0
    # Cropped[shape[0] - 4:shape[0], shape[1]-40:shape[1], :] = 0
    z = int(shape[0]*0.165)
    masked_out = np.zeros(shape)
    masked_out[5:shape[0] - 4, z:shape[1] - z, :] = scan_threshold[5:shape[0] - 4, 40:shape[1] - 40, :]
    FinalPred = masked_out + masked_out_pred
    FinalPred[FinalPred >= 0] = 1
    candidate_voxels = np.count_nonzero(FinalPred)
    voxel_vol = (voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]) / 3
    return FinalPred, candidate_voxels * voxel_vol
