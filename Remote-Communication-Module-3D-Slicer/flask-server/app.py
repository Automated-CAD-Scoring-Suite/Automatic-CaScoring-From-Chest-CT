import logging
import os
import sys
import time
from io import BytesIO
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from flask import Flask, request, after_this_request, jsonify, send_file

CurrentDir = os.path.dirname(os.path.realpath(__file__))
ParentDir = os.path.dirname(CurrentDir)
RepoRoot = os.path.dirname(ParentDir)
sys.path.append(RepoRoot)
from Models.crop_roi import get_coords, GetCoords
from Models.Segmentation.Inference import Infer

HeartTracePath = RepoRoot + "/Models/Segmentation/model_arch.pth"
HeartModelPath = RepoRoot + "/Models/Segmentation/HarD-MSEG-best.pth"

app = Flask(__name__)


def allow_CORS():
    @after_this_request
    def add_header(response):
        # To allow CORS (Cross Origin Resource Sharing)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response


@app.route('/')
def hello_world():
    allow_CORS()
    return 'Hello World!!!'


@app.route('/process')
def calculate_caScore():
    allow_CORS()
    print("Data Received..")

    print("Done.")
    return "CaScore: BlahBlah"


@app.route('/crop', methods=['POST'])
def GetCropCoordinates():
    if request.method == 'POST':
        # Get Slices Data & Shift Values from the request
        slices = request.files
        shift = request.form

        # Get Slices Segmentation
        SegmentedSlices = GetSlicesSegmentation(slices, shift)

        # Coordinates = GetCoords(SegmentedSlices, True)
        Coordinates = GetCoords(SegmentedSlices)

        # Send Coordinates
        return jsonify({"Coor": Coordinates})
    return 400


@app.route('/segment/slices', methods=['POST'])
def SegmentSlices():
    if request.method == 'POST':
        # Get Slices Data & Shift Values from the request
        slices = request.files
        shift = request.form

        if slices:
            # Get Slices Segmentation
            SegmentedSlices = GetSlicesSegmentation(slices, shift)

            # Compress For Sending
            CompressedArray = BytesIO()
            np.savez_compressed(CompressedArray, Ax=SegmentedSlices[0], Cor=SegmentedSlices[1], Sag=SegmentedSlices[2])
            CompressedArray.seek(0)

            # Send Segmented Slices
            return send_file(CompressedArray, attachment_filename="SegmentedSlices")
    return 400


@app.route('/segment/volume', methods=['POST'])
def SegmentVolume():
    if request.method == 'POST':
        # Get Compressed Volume
        VolumeCompressed = request.files["Volume"]

        # Decompress & Get Volume Array
        Data = np.load(VolumeCompressed)
        VolumeArray = Data['Volume']

        # Get Segmentation
        Segmentation = GetVolumeSegmentation(Volume=VolumeArray, ModelPath=HeartModelPath, TracePath=HeartTracePath)

        # Compress Segmentation Array
        CompressedArray = BytesIO()
        np.savez_compressed(CompressedArray, Segmentation=Segmentation)
        CompressedArray.seek(0)

        # Close The Loaded npz File To Prevent Memory Leaks
        Data.close()

        # Return The Segmented Volume
        return send_file(CompressedArray, attachment_filename="SegmentedVolume.npz")

    return "Good"


def GetSlicesSegmentation(Slices, Shift):
    Names = ["Ax1", "Ax2", "Ax3", "Sag1", "Sag2", "Sag3", "Cor1", "Cor2", "Cor3"]
    AxSlices = []
    SagSlices = []
    CorSlices = []
    SegmentedSlices = [[], [], []]
    model = Infer(trace_path=HeartTracePath, model_path=HeartModelPath,
                  axis=-1, slices=1)
    for SliceName in Names:
        Slice = Image.open(Slices[SliceName])
        SliceArray = np.array(Slice, dtype="int16")
        SliceArray += int(Shift[SliceName])
        res = model.predict(SliceArray)
        # plt.imshow(SliceArray, cmap='gray')
        # plt.show()
        # plt.imshow(res, cmap='gray')
        # plt.show()
        if "Ax" in SliceName:
            SegmentedSlices[0].append(res)
        elif "Sag" in SliceName:
            SegmentedSlices[1].append(res)
        elif "Cor" in SliceName:
            SegmentedSlices[2].append(res)
    return SegmentedSlices


def GetVolumeSegmentation(Volume, ModelPath, TracePath):
    Segmentation = []
    Times = []
    # Load Model
    model = Infer(trace_path=TracePath, model_path=ModelPath,
                  axis=-1, slices=1, shape=512)

    Start = time.time()

    # Loop Over Axial Slices
    for i in range(Volume.shape[0]):
        # Calculate Slice Time
        SliceStart = time.time()

        # Segment Heart in Slice
        res = model.predict(Volume[i, :, :])
        Segmentation.append(res)
        SliceEnd = time.time()
        SliceTime = (SliceEnd - SliceStart)
        print("Segmented Slice Number {} in {:.2f}".format(i, SliceTime))
        Times.append(SliceTime)

    End = time.time()
    print('Segmentation completed in {0:.2f} seconds'.format(End - Start))

    return Segmentation


if __name__ == '__main__':
    app.run(debug=True)
