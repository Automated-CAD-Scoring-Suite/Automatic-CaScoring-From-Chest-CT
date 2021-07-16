import logging
import os
import sys
import time
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, request, after_this_request, jsonify, send_file

CurrentDir = os.path.dirname(os.path.realpath(__file__))
ParentDir = os.path.dirname(CurrentDir)
RepoRoot = os.path.dirname(ParentDir)
sys.path.append(RepoRoot)
from Models.crop_roi import get_coords
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
def GetSlice():
    if request.method == 'POST':

        # Get Slices Data & Shift Values from the request
        slices = request.files
        shift = request.form

        if slices:
            # Get Slices Segmentation
            SegmentedSlices = GetSlicesSegmentation(slices, shift)

            # Get Coordinates For Each View
            AxCoor = [int(i) for i in get_coords(SegmentedSlices[0])]
            SagCoor = [int(i) for i in get_coords(SegmentedSlices[1])]
            CorCoor = [int(i) for i in get_coords(SegmentedSlices[2])]
            Coor = [AxCoor, SagCoor, CorCoor]

            # Send Coordinates
            return jsonify({"Coor": Coor})
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
            np.savez_compressed(CompressedArray, SegmentedSlices=SegmentedSlices)
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
    for SliceName in Names:
        Slice = Image.open(Slices[SliceName])
        SliceArray = np.array(Slice, dtype="int16")
        SliceArray += int(Shift[SliceName])
        model = Infer(trace_path=HeartTracePath, model_path=HeartModelPath,
                      axis=-1, slices=1, shape=512)
        res = model.predict(SliceArray)
        if "Ax" in SliceName:
            AxSlices.append(res)
        elif "Sag" in SliceName:
            SagSlices.append(res)
        elif "Cor" in SliceName:
            CorSlices.append(res)
    SegmentedSlices = [AxSlices, SagSlices, CorSlices]
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
