import os
import sys
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flask import Flask, request, after_this_request, jsonify, send_file

CurrentDir = os.path.dirname(os.path.realpath(__file__))
ParentDir = os.path.dirname(CurrentDir)
RepoRoot = os.path.dirname(ParentDir)
sys.path.append(RepoRoot)
from Models.crop_roi import get_coords
from Models.Segmentation.Inference import Infer

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
        # Names of Received Slices
        Names = ["Ax1", "Ax2", "Ax3", "Sag1", "Sag2", "Sag3", "Cor1", "Cor2", "Cor3"]
        # print(request.files)
        # Get first Axial Slice Data & Shift Value
        slices = request.files
        shift = request.form
        # Open Slice & reset shift
        if slices:
            AxSlices = []
            SagSlices = []
            CorSlices = []

            for SliceName in Names:
                Slice = Image.open(slices[SliceName])
                SliceArray = np.array(Slice, dtype="int16")
                SliceArray += int(shift[SliceName])
                if "Ax" in SliceName:
                    model = Infer(trace_path=RepoRoot + "/Models/Segmentation/model_arch.pth",
                                  model_path=RepoRoot + "/Models/Segmentation/HarD-MSEG-best.pth",
                                  axis=-1, slices=1, shape=512)
                    res = model.predict(SliceArray)
                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots(1, 1)
                    # ax[0][0].imshow(res, cmap='gray')
                    # plt.show()
                    AxSlices.append(res)
                # if "Ax" in SliceName:
                #     AxSlices.append(res)
                # elif "Sag" in SliceName:
                #     SagSlices.append(res)
                # elif "Cor" in SliceName:
                #     CorSlices.append(res)

            AxCoor = [int(i) for i in get_coords(AxSlices)]
            # SagCoor = [int(i) for i in get_coords(SagSlices)]
            # CorCoor = [int(i) for i in get_coords(CorSlices)]
            # Coor = [AxCoor, SagCoor, CorCoor]
            return jsonify({"Coor": AxCoor})
    return "Good"


@app.route('/segment/slices', methods=['POST'])
def SegmentSlices():
    if request.method == 'POST':
        # Names of Received Slices
        Names = ["Ax1", "Ax2", "Ax3", "Sag1", "Sag2", "Sag3", "Cor1", "Cor2", "Cor3"]
        # print(request.files)
        # Get first Axial Slice Data & Shift Value
        slices = request.files
        shift = request.form
        # Open Slice & reset shift
        if slices:
            AxSlices = []
            SagSlices = []
            CorSlices = []

            for SliceName in Names:
                Slice = Image.open(slices[SliceName])
                SliceArray = np.array(Slice, dtype="int16")
                SliceArray += int(shift[SliceName])
                if "Ax" in SliceName:
                    model = Infer(trace_path=RepoRoot + "/Models/Segmentation/model_arch.pth",
                                  model_path=RepoRoot + "/Models/Segmentation/HarD-MSEG-best.pth",
                                  axis=-1, slices=1, shape=512)
                    res = model.predict(SliceArray)
                    AxSlices.append(res)
                # if "Ax" in SliceName:
                #     AxSlices.append(res)
                # elif "Sag" in SliceName:
                #     SagSlices.append(res)
                # elif "Cor" in SliceName:
                #     CorSlices.append(res)
            CompressedArray = BytesIO()
            np.savez_compressed(CompressedArray, SegmentedSlices=AxSlices)
            CompressedArray.seek(0)
            # AxCoor = [int(i) for i in get_coords(AxSlices)]
            # SagCoor = [int(i) for i in get_coords(SagSlices)]
            # CorCoor = [int(i) for i in get_coords(CorSlices)]
            # Coor = [AxCoor, SagCoor, CorCoor]
            return send_file(CompressedArray, attachment_filename="SegmentedSlices")
    return "Good"


@app.route('/segment/volume', methods=['POST'])
def SegmentVolume():
    if request.method == 'POST':
        Segmentation = []

        # Load Model
        model = Infer(trace_path=RepoRoot + "/Models/Segmentation/model_arch.pth",
                      model_path=RepoRoot + "/Models/Segmentation/HarD-MSEG-best.pth",
                      axis=-1, slices=1, shape=512)

        # Get Compressed Volume
        VolumeCompressed = request.files["Volume"]

        # Decompress & Get Volume Array
        Data = np.load(VolumeCompressed)
        VolumeArray = Data['Volume']

        # Loop Over Axial Slices
        for i in range(VolumeArray.shape[0]):
            # Segment Heart in Slice
            res = model.predict(VolumeArray[i, :, :])
            Segmentation.append(res)

        # Compress Segmentation Array
        CompressedArray = BytesIO()
        np.savez_compressed(CompressedArray, Segmentation=Segmentation)
        CompressedArray.seek(0)

        # Close The Loaded npz File To Prevent Memory Leaks
        Data.close()

        return send_file(CompressedArray, attachment_filename="SegmentedVolume.npz")

    return "Good"


if __name__ == '__main__':
    app.run(debug=True)
