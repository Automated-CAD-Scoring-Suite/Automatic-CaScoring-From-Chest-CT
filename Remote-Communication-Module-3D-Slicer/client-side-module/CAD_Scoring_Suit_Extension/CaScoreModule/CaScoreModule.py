import logging
import os
import sys
import time
from io import BytesIO
import importlib
from distutils.util import strtobool

import numpy as np
import requests
import slicer
import vtk
import qt
from PIL import Image
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install

# Processing Packages

RepoRoot = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__))))))

sys.path.append(RepoRoot)

from Models.crop_roi import get_coords


#
# CaScoreModule
#

class CaScoreModule(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CaScoreModule"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#CaScoreModule">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
  Add data sets to Sample Data module.
  """
    # It is always recommended to provide sample data for users to make it easy to try the module, but if no sample
    # data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # TODO Add sample data for test

    # CaScoreModule1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='CaScoreModule',
        sampleName='CaScoreModule1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder. It can be
        # created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'CaScoreModule1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256"
             "/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='CaScoreModule1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='CaScoreModule1'
    )

    # CaScoreModule2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='CaScoreModule',
        sampleName='CaScoreModule2',
        thumbnailFileName=os.path.join(iconsPath, 'CaScoreModule2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256"
             "/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='CaScoreModule2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='CaScoreModule2'
    )


#
# CaScoreModuleWidget
#

class CaScoreModuleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent=None):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.LocalProcessing = True

    def setup(self):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/CaScoreModule.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = CaScoreModuleLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        # self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        # self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.OnlineProcessingRadio.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.LocalProcessingRadio.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.CroppingEnabled.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.PartialSegmentation.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.HeartSegNode.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.CalSegNode.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.SegAndCrop.toggled.connect(self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Radio Boxes
        self.ui.OnlineProcessingRadio.toggled.connect(self.ProcessingLocationSelect)
        self.ui.LocalProcessingRadio.toggled.connect(self.ProcessingLocationSelect)

        # Checkboxes
        self.ui.CroppingEnabled.toggled.connect(self.AllowableOperations)
        self.ui.PartialSegmentation.toggled.connect(self.AllowableOperations)
        self.ui.HeartSegNode.toggled.connect(self.AllowableOperations)
        self.ui.CalSegNode.toggled.connect(self.AllowableOperations)
        self.ui.SegAndCrop.toggled.connect(self.AllowableOperations)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
    Called when the application closes and the module widget is destroyed.
    """
        self.removeObservers()

    def enter(self):
        """
    Called each time the user opens this module.
    """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
    Called each time the user opens a different module.
    """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
    Called just before the scene is closed.
    """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
    Called just after the scene is closed.
    """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
    Ensure parameter node exists and observed.
    """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        # self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        # self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
        # self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        # self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

        self.ui.HeartModelPath.currentPath = self._parameterNode.GetParameter("HeartModelPath")
        self.ui.HeartTracePath.currentPath = self._parameterNode.GetParameter("HeartTracePath")
        self.ui.CalModelPath.currentPath = self._parameterNode.GetParameter("CalModelPath")
        self.ui.CalTracePath.currentPath = self._parameterNode.GetParameter("CalTracePath")

        if self._parameterNode.GetParameter("CroppingEnabled"):
            self.ui.CroppingEnabled.checked = strtobool(self._parameterNode.GetParameter("CroppingEnabled"))
            self.ui.PartialSegmentation.checked = strtobool(self._parameterNode.GetParameter("Partial"))
            self.ui.HeartSegNode.checked = strtobool(self._parameterNode.GetParameter("HeartSegNode"))
            self.ui.CalSegNode.checked = strtobool(self._parameterNode.GetParameter("CalSegNode"))
            self.ui.SegAndCrop.checked = strtobool(self._parameterNode.GetParameter("SegAndCrop"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume"):
            self.ui.applyButton.toolTip = "Compute CaScore"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input volume"
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        # self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)s
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        # self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)
        self._parameterNode.SetParameter("URL",
                                         self.ui.URLLineEdit.text if self.ui.URLLineEdit.isEnabled()
                                         else "http://localhost:5000")

        self._parameterNode.SetParameter("Local", "true" if self.ui.LocalProcessingRadio.checked else "false")
        self._parameterNode.SetParameter("Partial", "true" if self.ui.PartialSegmentation.checked else "false")
        self._parameterNode.SetParameter("HeartSegNode", "true" if self.ui.HeartSegNode.checked else "false")
        self._parameterNode.SetParameter("CalSegNode", "true" if self.ui.CalSegNode.checked else "false")
        self._parameterNode.SetParameter("CroppingEnabled", "true" if self.ui.CroppingEnabled.checked else "false")
        self._parameterNode.SetParameter("SegAndCrop", "true" if self.ui.SegAndCrop.checked else "false")
        self._parameterNode.SetParameter("Anonymize", "true" if self.ui.Anonymize.checked else "false")
        self._parameterNode.SetParameter("HeartModelPath", self.ui.HeartModelPath.currentPath)
        self._parameterNode.SetParameter("HeartTracePath", self.ui.HeartTracePath.currentPath)
        self._parameterNode.SetParameter("CalModelPath", self.ui.CalModelPath.currentPath)
        self._parameterNode.SetParameter("CalTracePath", self.ui.CalTracePath.currentPath)

        self._parameterNode.EndModify(wasModified)

    def ProcessingLocationSelect(self):
        """
        Handles Changes Processing Location Settings
        """

        if self.ui.LocalProcessingRadio.isChecked():
            self.ui.URLLineEdit.setDisabled(True)
            self.LocalProcessing = True
            self.ui.LocalSettings.setEnabled(True)
            self.ui.LocalSettings.collapsed = False
            self.ui.OnlineSettings.setEnabled(False)
            self.ui.OnlineSettings.collapsed = True

        elif self.ui.OnlineProcessingRadio.isChecked():
            self.ui.URLLineEdit.setEnabled(True)
            self.LocalProcessing = False
            self.ui.LocalSettings.setEnabled(False)
            self.ui.LocalSettings.collapsed = True
            self.ui.OnlineSettings.setEnabled(True)
            self.ui.OnlineSettings.collapsed = False

    def AllowableOperations(self):

        # Disable Partial Segmentation Option If Segmentation Node Creation Option is Enabled,
        # As We Need To Fully Segment The Heart, Also Disables Requesting Segmentation As It Is Required

        if strtobool(self._parameterNode.GetParameter("HeartSegNode")):
            self._parameterNode.SetParameter("Partial", "false")
            self.ui.PartialSegmentation.setEnabled(False)
            self._parameterNode.SetParameter("SegAndCrop", "false")
            self.ui.SegAndCrop.setEnabled(False)
        else:
            self.ui.PartialSegmentation.setEnabled(True)
            self.ui.SegAndCrop.setEnabled(True)

        # Disable Partial Segmentation Option If Cropping is Disabled
        if strtobool(self._parameterNode.GetParameter("CroppingEnabled")) and \
                not strtobool(self._parameterNode.GetParameter("HeartSegNode")):
            self.ui.PartialSegmentation.setEnabled(True)
        else:
            self._parameterNode.SetParameter("Partial", "false")
            self.ui.PartialSegmentation.setEnabled(False)

        self.updateGUIFromParameterNode()

    def onApplyButton(self):
        """
    Run processing when user clicks "Apply" button.
    """
        startTime = time.time()
        logging.info('Processing started')

        # Collapse Settings For Better Progress View
        self.ui.GeneralSettings.collapsed = True
        self.ui.LocalSettings.collapsed = True
        self.ui.OnlineSettings.collapsed = True

        # Enable & Expand Progress Box
        self.ui.Progress.setEnabled(True)
        self.ui.Progress.collapsed = False

        # Update Parameters
        self.updateParameterNodeFromGUI()

        # Get Parameters
        Partial = bool(strtobool(self._parameterNode.GetParameter("Partial")))
        HeartSegNode = bool(strtobool(self._parameterNode.GetParameter("HeartSegNode")))
        CalSegNode = bool(strtobool(self._parameterNode.GetParameter("CalSegNode")))
        CroppingEnabled = bool(strtobool(self._parameterNode.GetParameter("CroppingEnabled")))
        SegAndCrop = bool(strtobool(self._parameterNode.GetParameter("SegAndCrop")))
        HeartModelPath = self._parameterNode.GetParameter("HeartModelPath")
        HeartTracePath = self._parameterNode.GetParameter("HeartTracePath")
        CalModelPath = self._parameterNode.GetParameter("CalModelPath")
        CalTracePath = self._parameterNode.GetParameter("CalTracePath")

        # Get Input Volume
        InputVolumeNode = self.ui.inputSelector.currentNode()

        try:

            # Initialize Variables
            Segmentation = []
            SegmentationTime = 0
            Coordinates = []
            VolumeArray = np.array(slicer.util.arrayFromVolume(InputVolumeNode), copy=True)

            # Check For Dependencies & Install Missing Ones
            if self.LocalProcessing:
                self.logic.CheckDependencies()

            # Compute output
            if SegAndCrop and not self.LocalProcessing:
                Coordinates = self.logic.SegAndCrop(VolumeArray, self.LocalProcessing,
                                                    self.ui.URLLineEdit.text, HeartModelPath, HeartTracePath)

            elif CalSegNode or CroppingEnabled:
                Segmentation, SegmentationTime = self.logic.Segment(VolumeArray, self.LocalProcessing,
                                                                    self.ui.URLLineEdit.text, Partial, True,
                                                                    HeartModelPath, HeartTracePath)

                logging.info('Segmentation completed in {0:.2f} seconds'.format(SegmentationTime))

            if not Partial and CalSegNode:
                self.logic.CreateSegmentationNode(Segmentation, "Heart")

            if CroppingEnabled and not SegAndCrop:
                Coordinates = self.logic.GetCoordinates(Segmentation, Partial, self.LocalProcessing)

            if CroppingEnabled or SegAndCrop:
                x1 = (Coordinates[0] - 20) if (Coordinates[0] - 20 >= 0) else 0
                x2 = (Coordinates[1] + 20) if (Coordinates[1] + 20 >= 0) else VolumeArray.shape[1]
                y1 = (Coordinates[2] - 20) if (Coordinates[2] - 20 >= 0) else 0
                y2 = (Coordinates[3] + 20) if (Coordinates[3] + 20 >= 0) else VolumeArray.shape[3]

                logging.info(f"The Cropping Coordinates Are X->{x1}:{x2}, Y->{y1}:{y2}")
                NewVolume = VolumeArray[:, x1:x2, y1:y2]
                slicer.util.updateVolumeFromArray(InputVolumeNode, NewVolume)
                logging.info(f"Cropped!")
                CompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
                VolumeNodeID = CompositeNode.GetBackgroundVolumeID()
                CurrentNode = slicer.mrmlScene.GetNodeByID(VolumeNodeID)
                slicer.util.setSliceViewerLayers(foreground=CurrentNode, fit=True)

        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: " + str(e))
            import traceback
            traceback.print_exc()

        stopTime = time.time()
        logging.info('Processing completed in {0:.2f} seconds'.format(stopTime - startTime))


#
# CaScoreModuleLogic
#

class CaScoreModuleLogic(ScriptedLoadableModuleLogic, qt.QObject):
    """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
    finished = qt.Signal()
    progress = qt.Signal(int)

    def __init__(self):
        """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
    Initialize parameter node with default settings.
    """
        if not parameterNode.GetParameter("URL"):
            parameterNode.SetParameter("URL", "http://localhost:5000")
        if not parameterNode.GetParameter("Local"):
            parameterNode.SetParameter("Local", "true")
        if not parameterNode.GetParameter("HeartModelPath"):
            Path = RepoRoot + '/Models/Segmentation/HarD-MSEG-best.pth'
            if os.path.exists(Path):
                parameterNode.SetParameter("HeartModelPath", Path)
        if not parameterNode.GetParameter("HeartTracePath"):
            Path = RepoRoot + '/Models/Segmentation/model_arch.pth'
            if os.path.exists(Path):
                parameterNode.SetParameter("HeartTracePath", Path)

    def processOld(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True,
                                 update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info('Processing completed in {0:.2f} seconds'.format(stopTime - startTime))

    def SegAndCrop(self, inputVolume, LocalProcessing=True, ProcessingURL="http://localhost:5000",
                   ModelPath="", TracePath=""):

        if inputVolume is None:
            raise ValueError("Input volume is invalid")

        startTime = time.time()
        logging.info('Processing started')

        # Convert Volume To NumPy Array
        VolumeArray = np.copy(inputVolume)
        VolumeShape = VolumeArray.shape

        # Cropping Pattern Start
        # CompressedArray = BytesIO()
        # np.savez_compressed(CompressedArray, a=VolumeArray)

        # Axial, Sagittal, Coronal
        Names = ["Ax1", "Ax2", "Ax3", "Sag1", "Sag2", "Sag3", "Cor1", "Cor2", "Cor3"]
        files = {}
        ShiftValues = {}
        RawSliceArrays = [[], [], []]
        Arr = []
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
                    # Shift Array Values if There Exists -Ve Values, since -ve values are lost during PNG conversion,
                    # and store the shift value to be sent
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
            SliceSendReq = requests.post(ProcessingURL + "/crop", files=files, data=ShiftValues)
            Coordinates = SliceSendReq.json()["Coor"]
            logging.info(f"Received Cropping Coordinates From Online Server")
        else:
            from Models.Segmentation.Inference import Infer
            model = Infer(trace_path=TracePath, model_path=ModelPath)
            res = model.predict(np.array(RawSliceArrays[0]))
            Coordinates.append(get_coords(res))

            # for x in range(0, 3):
            #     print(x)
            #     Coordinates.append(get_coords(RawSliceArrays[x]))

            logging.info(f"Cropping Coordinates Calculated Locally")

        logging.info(f"The Cropping Coordinates Are {Coordinates}")
        stopTime = time.time()
        logging.info(
            'Segmentation & Coordinates Calculation Completed in in {0:.2f} seconds'.format(stopTime - startTime))

        return Coordinates
        # [z,x,y]
        # Coordinates = [[Xmin, Xmax, Ymin, Ymax],[Zmin,Zmax, Ymin, Ymax],[Zmin, Zmax, Xmin,Xmax]]
        # Start Cropping
        # Determine Correct Cropping Coordinates

        # x1 = np.minimum(Coordinates[0][0], Coordinates[2][0])
        # x2 = np.maximum(Coordinates[0][1], Coordinates[2][1])
        # y1 = np.minimum(Coordinates[0][2], Coordinates[1][0])
        # y2 = np.maximum(Coordinates[0][3], Coordinates[1][1])
        # z1 = np.minimum(Coordinates[1][2], Coordinates[2][2])
        # z2 = np.maximum(Coordinates[1][3], Coordinates[2][3])
        # logging.info(f"The Cropping Coordinates Are X->{x1}:{x2}, Y->{y1}:{y2}, Z->{z1}:{z2}")

        # LabelVolume Tests
        # imageOrigin = [0.0, 0.0, 0.0]
        # imageSpacing = [0.4883, 0.4883, 2.5]
        # imageDirections = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        # a = np.zeros([VolumeShape[0], VolumeShape[1], VolumeShape[2]])
        # a[0:VolumeShape[0], 0:VolumeShape[1], 0:VolumeShape[2]] = 1
        # print(a)
        # LabelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'Heart-Label')
        # LabelmapVolumeNode.SetOrigin(imageOrigin)
        # LabelmapVolumeNode.SetSpacing(imageSpacing)
        # LabelmapVolumeNode.SetIJKToRASDirections(imageDirections)
        # slicer.util.updateVolumeFromArray(LabelmapVolumeNode, a)
        # LabelmapVolumeNode.CreateDefaultDisplayNodes()
        # LabelmapVolumeNode.CreateDefaultStorageNode()
        # slicer.util.loadLabelVolume(r'c:\Users\msliv\Documents\test.nrrd')

    def Segment(self, inputVolume, LocalProcessing=True, ProcessingURL="http://localhost:5000", Partial=True,
                ReturnTime=True, ModelPath="", TracePath=""):

        if inputVolume is None:
            raise ValueError("Input volume is invalid")

        # Get Segmentation Start Time
        SegmentStart = time.time()

        # Convert Volume To NumPy Array
        VolumeArray = np.copy(inputVolume)
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

        if ReturnTime:
            return SegmentedSlices, SegmentTime
        else:
            return SegmentedSlices

    def CheckDependencies(self):

        # Install PyTorch if Not Detected
        Torch = importlib.util.find_spec("torch")

        if Torch is None:
            logging.info('Installing PyTorch')
            pip_install("torch")
            logging.info('PyTorch Installed')
        else:
            logging.info('PyTorch Found')

    def CreateSegmentationNode(self, Segmentation, Name="Heart"):

        # Create a new LabelMapVolume
        LabelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', f'{Name}-Label')

        # Update the LabelMapVolume from the given Segmentation array
        slicer.util.updateVolumeFromArray(LabelMapVolumeNode, Segmentation)

        # Create a SegmentationNode
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f'{Name}-Segmentation')

        # Load the LabelMapVolume into the SegmentationNode
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(LabelMapVolumeNode, segNode)

        # Update Display
        # LabelmapVolumeNode.CreateDefaultDisplayNodes()
        # LabelmapVolumeNode.CreateDefaultStorageNode()

    def GetCoordinates(self, Segmentation, Partial, Local):
        Coordinates = []
        if Partial:
            Coordinates = get_coords(Segmentation)
        else:
            Z = (Segmentation.shape[0]) / 2
            Coordinates = get_coords(Segmentation[Z - 1:Z + 1, :, :])

        return Coordinates


#
# CaScoreModuleTest
#

class CaScoreModuleTest(ScriptedLoadableModuleTest):
    """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
    """
        self.setUp()
        self.test_CaScoreModule1()

    def test_CaScoreModule1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('CaScoreModule1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = CaScoreModuleLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
