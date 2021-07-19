import numpy as np
import time
import requests
from io import BytesIO
import logging
from PIL import Image
import os
import sys
import pickle

RepoRoot = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.realpath(__file__))))))))

sys.path.append(RepoRoot)

with open('data.pkl', 'rb') as f:
    Input = pickle.load(f)

VolumeArray = Input["VolumeArray"]
ModelPath = Input["ModelPath"]
Local = Input["Local"]
ServerURL = Input["ServerURL"]
Partial = Input["Partial"]


def GetSampleSlicesFromVolume(VolumeArray=None, Local=True):
    # Axial, Sagittal, Coronal
    Names = ["Ax1", "Ax2", "Ax3", "Sag1", "Sag2", "Sag3", "Cor1", "Cor2", "Cor3"]
    files = {}
    ShiftValues = {}
    RawSliceArrays = [[], [], []]
    VolumeShape = VolumeArray.shape

    # Prepare Slices
    for i in range(3):
        Mid = int(VolumeShape[i] / 2)
        if i == 0:
            logging.info(f"Preparing Axial Slices Number {Mid - 1}, {Mid}, {Mid + 1}")
        elif i == 1:
            logging.info(f"Preparing Sagittal Slices Number {Mid - 1}, {Mid}, {Mid + 1}")
        elif i == 2:
            logging.info(f"Preparing Coronal Slices Number {Mid - 1}, {Mid}, {Mid + 1}")
        for j in range(-1, 2):
            if i == 0:
                # Prepare Axial Slices
                Arr = VolumeArray[Mid + j, :, :]
            elif i == 1:
                # Prepare Sagittal Slices
                Arr = VolumeArray[:, Mid + j, :]
            elif i == 2:
                # Prepare Coronal Slices
                Arr = VolumeArray[:, :, Mid + j]
            RawSliceArrays[i].append(Arr)
            if not Local:
                ArrS = Arr.min()
                # Shift Array Values if There Exists -Ve Values, since -ve values are lost during PNG
                # conversion, and store the shift value to be sent
                if ArrS < 0:
                    Arr -= ArrS
                    ShiftValues[Names[0]] = ArrS
                else:
                    ShiftValues[Names[0]] = 0
                SliceImg = Image.fromarray(Arr)
                SliceBytes = BytesIO()
                SliceImg.save(SliceBytes, format="PNG")
                SliceBytes.seek(0, 0)
                files[Names.pop(0)] = SliceBytes

    return RawSliceArrays, files, ShiftValues


# TODO: Receive Routes From Caller

startTime = time.time()
logging.info('Processing started')

# Prepare Slices
RawSliceArrays, files, ShiftValues = GetSampleSlicesFromVolume(VolumeArray=VolumeArray,
                                                               Local=Local)

# Send to Server For Processing
SliceSendReq = requests.post(ServerURL + "/crop", files=files, data=ShiftValues)
Coordinates = SliceSendReq.json()["Coor"]
logging.info(f"Received Cropping Coordinates From Online Server")

logging.info(f"The Cropping Coordinates Are {Coordinates}")
stopTime = time.time()
SegAndCropTime = stopTime - startTime

logging.info(
    'Segmentation & Coordinates Calculation Completed in {0:.2f} seconds'.format(SegAndCropTime))

output = {'Coordinates': Coordinates, 'SegAndCropTime': SegAndCropTime}

sys.stdout.buffer.write(pickle.dumps(output))
