from flask import Flask, request, after_this_request, flash, redirect, url_for
import numpy as np
from PIL import Image

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
        # Names = ["Ax1", "Ax2", "Ax3", "Sag1", "Sag2", "Sag3", "Cor1", "Cor2", "Cor3"]
        # print(request.files)
        # Get first Axial Slice Data & Shift Value
        file = request.files['Ax1']
        shift = request.form['Ax1']
        # print(request.form)
        # Open Slice & reset shift
        # print(file)
        if file:
            Slice = Image.open(file)
            SliceArray = np.array(Slice, dtype="int16")
            # print(SliceArray)
            SliceArray += int(shift)
            # print(SliceArray)
            # print(file.filename)
            # print(shift)
            # print(SliceArray.shape)
            # print(SliceArray.max())
            # print(SliceArray.min())
    return "Good"


if __name__ == '__main__':
    app.run(debug=True)
