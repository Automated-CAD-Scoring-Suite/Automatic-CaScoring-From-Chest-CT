#!/usr/bin/env python-real

import os
import sys
import time
import logging
import numpy as np
from io import BytesIO
from PIL import Image
import requests

RepoRoot = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__))))))

sys.path.append(RepoRoot)


def main(inputVolume, LocalProcessing=True, ProcessingURL="http://localhost:5000", Partial=True,
         ModelPath="", TracePath="", outputVolume=None):
    # TODO: Receive Routes From Caller
    if outputVolume is None:
        outputVolume = []

    if inputVolume is None:
        raise ValueError("Input volume is invalid")

    # Get Segmentation Start Time
    SegmentStart = time.time()

    # Convert Volume To NumPy Array
    Data = np.load(inputVolume.encode('UTF-8', errors="ignore"))
    VolumeArray = Data['Volume']
    VolumeShape = VolumeArray.shape
    SegmentedSlices = []

    # Segment 3 Slicers From Each View
    if Partial:
        # Axial, Sagittal, Coronal
        Names = ["Ax1", "Ax2", "Ax3", "Sag1", "Sag2", "Sag3", "Cor1", "Cor2", "Cor3"]
        files = {}
        ShiftValues = {}
        RawSliceArrays = [[], [], []]
        Coordinates = []

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
                if not LocalProcessing:
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

        if not LocalProcessing:
            SliceSendReq = requests.post(ProcessingURL + "/segment/slices", files=files, data=ShiftValues)
            Response = BytesIO(SliceSendReq.content)
            Response.seek(0)
            Data = np.load(Response)
            SegmentedSlices = np.copy(Data["SegmentedSlices"])
            Data.close()
            logging.info(f"Segmented Slices Received From Server")
            # logging.info(f"Received Cropping Coordinates From Online Server")
        else:
            from Models.Segmentation.Inference import Infer
            model = Infer(trace_path=TracePath,
                          model_path=ModelPath,
                          axis=-1, slices=1, shape=512)

            for slice in RawSliceArrays[0]:
                SegmentedSlices.append(model.predict(np.array(slice)))
            # for x in range(0, 3):
            #     print(x)
            #     Coordinates.append(get_coords(RawSliceArrays[x]))

            # Coordinates.append(get_coords(res))
            logging.info(f"Segmentation Computed Locally")
            # logging.info(f"Cropping Coordinates Calculated Locally")
    else:
        if not LocalProcessing:
            CompressedVolume = BytesIO()
            np.savez_compressed(CompressedVolume, Volume=VolumeArray)
            CompressedVolume.seek(0)
            SliceSendReq = requests.post(ProcessingURL + "/segment/volume", files={"Volume": CompressedVolume})
            Response = BytesIO(SliceSendReq.content)
            Response.seek(0)
            Data = np.load(Response)
            SegmentedSlices = np.copy(Data['Segmentation'])
            Data.close()
            logging.info(f"Segmented Slices Received From Server")
            # logging.info(f"Received Cropping Coordinates From Online Server")
        else:
            from Models.Segmentation.Inference import Infer
            model = Infer(trace_path=TracePath, model_path=ModelPath,
                          axis=-1, slices=1, shape=512)

            for i in range(VolumeShape[0]):
                # Segment Heart in Slice
                res = model.predict(VolumeArray[i, :, :])
                SegmentedSlices.append(res)
            # SegmentedSlices = model.predict(np.asarray(RawSliceArrays[0]))
            # for x in range(0, 3):
            #     print(x)
            #     Coordinates.append(get_coords(RawSliceArrays[x]))
            # Coordinates.append(get_coords(res))
            logging.info(f"Segmentation Computed Locally")
            # logging.info(f"Cropping Coordinates Calculated Locally")

    # Calculate Segmentation Time
    SegmentEnd = time.time()
    SegmentTime = SegmentEnd - SegmentStart

    CompressedVolume = BytesIO()
    np.savez_compressed(CompressedVolume, Segmentation=SegmentedSlices)
    CompressedVolume.seek(0)
    VolumeString = CompressedVolume.read()
    Data.close()
    outputVolume = VolumeString.decode('UTF-8', errors="ignore")
    sys.argv[7] = outputVolume


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: SegmentationFromModel <input> <sigma> <output>")
        sys.exit(1)
    main(sys.argv[1], bool(sys.argv[2]), sys.argv[3], bool(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7])
