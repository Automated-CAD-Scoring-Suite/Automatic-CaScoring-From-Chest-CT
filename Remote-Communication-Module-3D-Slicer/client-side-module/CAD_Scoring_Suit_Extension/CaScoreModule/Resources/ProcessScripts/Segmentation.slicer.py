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

# Get Segmentation Start Time
SegmentStart = time.time()

# Convert Volume To NumPy Array
VolumeShape = VolumeArray.shape
SegmentedSlices = []

# Segment 3 Slicers From Each View
if Partial:
    SegmentedSlices = [[], [], []]
    RawSliceArrays, files, ShiftValues = GetSampleSlicesFromVolume(VolumeArray=VolumeArray,
                                                                   Local=Local)

    if not Local:
        # Send Data To Server For Processing
        SliceSendReq = requests.post(ServerURL + "/segment/slices", files=files, data=ShiftValues)
        Response = BytesIO(SliceSendReq.content)
        Response.seek(0)
        Data = np.load(Response)
        SegmentedSlices[0] = np.copy(Data["Ax"])
        SegmentedSlices[1] = np.copy(Data["Cor"])
        SegmentedSlices[2] = np.copy(Data["Sag"])
        Data.close()
        logging.info(f"Segmented Slices Received From Server")

    else:
        # Load Model
        from Models.Segmentation.Inference import Infer

        model = Infer(model_path=ModelPath, model_input=(112, 112, 112))
        # Loop over 3 slices in each View and apply heart segmentation
        for i in range(3):
            SegSlice = model.predict(np.array(RawSliceArrays[i]))
            SegmentedSlices[i].append(SegSlice)

        logging.info(f"Segmentation Computed Locally")

else:
    if not Local:
        CompressedVolume = BytesIO()
        np.savez_compressed(CompressedVolume, Volume=VolumeArray)
        CompressedVolume.seek(0)
        SliceSendReq = requests.post(ServerURL + "/segment/volume", files={"Volume": CompressedVolume})
        Response = BytesIO(SliceSendReq.content)
        Response.seek(0)
        Data = np.load(Response)
        SegmentedSlices = np.copy(Data['Segmentation'])
        Data.close()
        logging.info(f"Segmented Slices Received From Server")

    else:
        from Models.Segmentation.Inference import Infer

        model = Infer(model_path=ModelPath, model_input=(112, 112, 112))
        # Calculate Slice Time
        SliceStart = time.time()

        SegmentedSlices = model.predict(VolumeArray)

        SliceEnd = time.time()
        SliceTime = (SliceEnd - SliceStart)
        print("Segmented The Volume in {:.2f}".format(SliceTime))

        logging.info(f"Segmentation Computed Locally")

# Calculate Segmentation Time
SegmentEnd = time.time()
SegmentTime = SegmentEnd - SegmentStart

output = {'Segmentation': SegmentedSlices, 'SegmentationTime': SegmentTime}

sys.stdout.buffer.write(pickle.dumps(output))
