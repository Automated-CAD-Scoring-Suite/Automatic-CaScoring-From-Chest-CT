# Coronary Artery Disease Scoring Suit For 3D Slicer
# Remote-Communication-Module-for-3D-Slicer

```
Insert a suitable Intro here
```

A loadable module for 3D Slicer Platform written 
in python, with a flask server for easy deployment 
on cloud computing services.


## Requirements

### For The Extension
`scikit-image tensorflow`

*Installed Automatically By The Extension In Slicer's 
Packaged Python*

The Slicer Module [SlicerProcesses](https://github.com/pieper/SlicerProcesses) is required 
and comes pre-packaged with the extension

### For The Server
`flask scikit-image tensorflow pillow`

A requirements file for easy setup is included and can be found 
[here](flask-server/requirements.txt).

# The Main Module's Features

Our Main Module, The **CaScoreModule** has two main 
operation modes:

1. Local Mode: The data is processed locally on the user's machine,
note that for optimal performance, a system capable of running
   TensorFlow Models is needed, A CUDA-Capable GPU will increase performance
   drastically, TensorFlow and other required packages needs to be installed 
   inside Slicer's Packaged Python, A check will be done during first operation
   in this mode and all the required packages will be automatically installed.
   
2. Online Mode: The data is processed online, the volume is sent to a given URL
which will process the data and send it back, an example server is included in
   this repo together with a requirements file.
   
The Module's widget is divided into 6 main sections which helps
the user in selecting the settings they need and also see the progress &
the results. These sections are:

## Input Volume Select

![Input Volume Select](Images/Volume%20Select.png)

In this section, we simply select which one of the loaded 
volumes will we do the processing on

## General Settings

![General Settings](Images/General%20Settings.png)

In this section, we select the main parameters for our processing:

### Processing Location
We have two main modes in our module, a local mode and an online mode,
This setting determines which one we will use, in summary the difference
between them is:

1. Local Mode: The data is processed locally on the user's machine,
   note that for optimal performance, a system capable of running
   TensorFlow Models is needed, A CUDA-Capable GPU will increase performance
   drastically, TensorFlow and other required packages needs to be installed 
   inside Slicer's Packaged Python, A check will be done during first operation
   in this mode and all the required packages will be automatically installed.
   
2. Online Mode: The data is processed online, the volume is sent to a given URL
   which will process the data and send it back, an example server is included in
   this repo together with a requirements file.
   
### Cropping

This setting enables cropping, it uses the results of a Deep Learning
model built using TensorFlow to segment the heart, determining its location,
then we create a bounding box around the ROI (The Heart) predicted by the model and
crop out everything outside it.

### Partial Segmentation

This enables the Partial Segmentation mode, in this mode we select the three middle
slices from the Axial, Sagittal & Coronal views and use them to get the bounding box's 
coordinates used in cropping, while it's not as accurate or useful as segmenting 
the whole volume, it's role shines as a pre-processing step, this rough cropping 
could be done before segmenting the whole volume to increase its quality, or decrease
the volume's size which is useful in cases of using the online mode in a low or limited
internet bandwidth setting.

### Create A Heart Segmentation Node

Create a Segmentation node from the results of the full heart segmentation, enabling this 
disables the partial segmentation option, this segmentation is shown over the input heart
CT image and shows the location of the heart, it could also be converted into a closed
surface representation of the heart. This option is required to find calcifications.

### Visualize The Heart As A Closed Surface

Creates a closed surface representation of the heart's segmentation, this creates a 3D view
of the heart which could have numerous benefits, this option requires the full volume 
segmentation and the creation of a segmentation node.

### Find Calcifications And Create A Segmentation Node

Uses one of two available methods to find calcifications in the heart and save it in a 
segmentation node, by default uses Image Processing techniques to determine the calcification
the calculates their total volume. Requires full heart segmentation.

### Visualize The Heart As A Closed Surface

Creates a closed surface representation of the calcifications, similar to that of the heart.




