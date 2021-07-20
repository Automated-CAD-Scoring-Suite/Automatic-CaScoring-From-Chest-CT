import logging
import os
import random
import sys
import time
from io import BytesIO
import importlib
from distutils.util import strtobool

import numpy as np
import requests
from requests.exceptions import ConnectionError
import slicer
import vtk
import qt
from PIL import Image
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install
from Processes import Process, ProcessesLogic
import pickle

# Processing Packages

RepoRoot = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__))))))

sys.path.append(RepoRoot)

from Models.crop_roi import get_coords, GetCoords


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
    """
    Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self.logic = None  # Used to store Logic Class
        self._parameterNode = None  # Used to store the parameter node object
        self._updatingGUIFromParameterNode = False

        self.LocalProcessing = True  # Flag to check if the Processing is Offline (Local) or Online

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
        self.ui.HeartSeg3D.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.CalSegNode.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.CalSeg3D.toggled.connect(self.updateParameterNodeFromGUI)
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
        self.ui.HeartSeg3D.toggled.connect(self.AllowableOperations)
        self.ui.CalSegNode.toggled.connect(self.AllowableOperations)
        self.ui.CalSeg3D.toggled.connect(self.AllowableOperations)
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
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
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
        self.ui.CalModelPath.currentPath = self._parameterNode.GetParameter("CalModelPath")

        if self._parameterNode.GetParameter("CroppingEnabled"):
            self.ui.CroppingEnabled.checked = strtobool(self._parameterNode.GetParameter("CroppingEnabled"))
            self.ui.PartialSegmentation.checked = strtobool(self._parameterNode.GetParameter("Partial"))
            self.ui.HeartSegNode.checked = strtobool(self._parameterNode.GetParameter("HeartSegNode"))
            self.ui.HeartSeg3D.checked = strtobool(self._parameterNode.GetParameter("HeartSeg3D"))
            self.ui.CalSegNode.checked = strtobool(self._parameterNode.GetParameter("CalSegNode"))
            self.ui.CalSeg3D.checked = strtobool(self._parameterNode.GetParameter("CalSeg3D"))
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
        self._parameterNode.SetParameter("URL",
                                         self.ui.URLLineEdit.text if self.ui.URLLineEdit.isEnabled()
                                         else "http://localhost:5000")

        self._parameterNode.SetParameter("Local", "true" if self.ui.LocalProcessingRadio.checked else "false")
        self._parameterNode.SetParameter("Partial", "true" if self.ui.PartialSegmentation.checked else "false")
        self._parameterNode.SetParameter("HeartSegNode", "true" if self.ui.HeartSegNode.checked else "false")
        self._parameterNode.SetParameter("HeartSeg3D", "true" if self.ui.HeartSeg3D.checked else "false")
        self._parameterNode.SetParameter("CalSegNode", "true" if self.ui.CalSegNode.checked else "false")
        self._parameterNode.SetParameter("CalSeg3D", "true" if self.ui.CalSeg3D.checked else "false")
        self._parameterNode.SetParameter("CroppingEnabled", "true" if self.ui.CroppingEnabled.checked else "false")
        self._parameterNode.SetParameter("SegAndCrop", "true" if self.ui.SegAndCrop.checked else "false")
        self._parameterNode.SetParameter("Anonymize", "true" if self.ui.Anonymize.checked else "false")
        self._parameterNode.SetParameter("HeartModelPath", self.ui.HeartModelPath.currentPath)
        self._parameterNode.SetParameter("CalModelPath", self.ui.CalModelPath.currentPath)

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

        self.AllowableOperations()

    def AllowableOperations(self):

        # Disable Partial Segmentation Option If Segmentation Node Creation Option is Enabled,
        # As We Need To Fully Segment The Heart, Also Disables Requesting Segmentation As It Is Required

        if strtobool(self._parameterNode.GetParameter("HeartSegNode")):
            self._parameterNode.SetParameter("Partial", "false")
            self.ui.PartialSegmentation.setEnabled(False)
            self._parameterNode.SetParameter("SegAndCrop", "false")
            self.ui.SegAndCrop.setEnabled(False)
            self.ui.HeartSeg3D.setEnabled(True)
        else:
            self.ui.PartialSegmentation.setEnabled(True)
            self.ui.SegAndCrop.setEnabled(True)
            self.ui.HeartSeg3D.setEnabled(False)
            self._parameterNode.SetParameter("HeartSeg3D", "false")

        # Disable Partial Segmentation Option If Cropping is Disabled
        if strtobool(self._parameterNode.GetParameter("CroppingEnabled")) and \
                not strtobool(self._parameterNode.GetParameter("HeartSegNode")):

            self.ui.PartialSegmentation.setEnabled(True)

            # Disable Don't Request Segmentation if Not Using Partial Segmentation
            if strtobool(self._parameterNode.GetParameter("Partial")):
                self.ui.SegAndCrop.setEnabled(True)
            else:
                self.ui.SegAndCrop.setEnabled(False)
        else:
            self._parameterNode.SetParameter("Partial", "false")
            self.ui.PartialSegmentation.setEnabled(False)

        self.updateGUIFromParameterNode()

    def RecenterVolume(self):
        # Center Volume in the Scene an update the view
        CompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
        VolumeNodeID = CompositeNode.GetBackgroundVolumeID()
        CurrentNode = slicer.mrmlScene.GetNodeByID(VolumeNodeID)
        slicer.util.setSliceViewerLayers(foreground=CurrentNode, fit=True)

    def reportProgress(self, Progress):
        logging.info(Progress)

    def ProcessingCompleted(self):
        self.ui.applyButton.setEnabled(True)

        # Recenter & Fit The Volume
        self.RecenterVolume()

        stopTime = time.time()
        logging.info('Processing completed in {0:.2f} seconds'.format(stopTime - self.startTime))

    def ProcessingStarted(self):
        # Collapse Settings For Better Progress View
        self.ui.GeneralSettings.collapsed = True
        self.ui.LocalSettings.collapsed = True
        self.ui.OnlineSettings.collapsed = True

        # Enable & Expand Progress Box
        self.ui.Progress.setEnabled(True)
        self.ui.Progress.collapsed = False

        # Disable Apply Button
        self.ui.applyButton.setDisabled(True)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        # Update Parameters
        self.updateParameterNodeFromGUI()

        # Get Input Volume
        InputVolumeNode = self.ui.inputSelector.currentNode()

        # Update GUI
        self.ProcessingStarted()

        self.startTime = time.time()
        logging.info('Processing started')

        try:

            self.logic.StartOperations(InputVolumeNode, self._parameterNode, self.reportProgress,
                                       self.ProcessingCompleted)

        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: " + str(e))
            import traceback
            traceback.print_exc()


# Helper Classes
class Signals(qt.QObject):
    finished = qt.Signal()
    progress = qt.Signal(str)


# Processes Classes
class SegmentationProcess(Process):

    def __init__(self, scriptPath, VolumeArray, Local, ServerURL, Partial, ModelPath):
        Process.__init__(self, scriptPath)
        self.VolumeArray = VolumeArray  # Numpy array, to use as input for the model.
        self.ModelPath = ModelPath  # Path to the TF model you'd like to load, as TF Models are not picklable.
        self.Local = Local
        self.ServerURL = ServerURL
        self.Partial = Partial
        self.Name = f"Segmentation-{os.path.basename(ModelPath)}"
        self.Output = None
        self.Segmentation = None
        self.SegmentationTime = None

    def prepareProcessInput(self):
        InputData = {
            'VolumeArray': self.VolumeArray,
            'ModelPath': self.ModelPath,
            'Local': self.Local,
            'ServerURL': self.ServerURL,
            'Partial': self.Partial,
        }
        with open('data.pkl', 'wb') as f:
            pickle.dump(InputData, f)

    def useProcessOutput(self, processOutput):
        output = pickle.loads(processOutput)
        os.remove('data.pkl')
        self.Output = output


#
# CaScoreModuleLogic
#

class CaScoreModuleLogic(ScriptedLoadableModuleLogic):
    """
    This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
        ScriptedLoadableModuleLogic.__init__(self)
        self.FinishedCallback = None
        self.UpdateCallback = None
        self.Local = None
        self.Partial = None
        self.HeartSegNode = None
        self.HeartSeg3D = None
        self.CalSegNode = None
        self.CalSeg3D = None
        self.CroppingEnabled = None
        self.SegAndCrop = None
        self.HeartModelPath = None
        self.CalModelPath = None
        self.ServerURL = None
        self.SegAndCropDone = None
        self.HeartSegDone = None
        self.HeartSegNodeDone = None
        self.CoordinatesCalculated = None
        self.Segmentation = None
        self.SegmentationTime = None
        self.Coordinates = None
        self.VolumeIJKToRAS = None
        self.VolumeArray = None
        self.InputVolumeNode = None
        self.DependenciesChecked = None
        self.CroppingDone = None
        self.HeartSegmentationProcess = None
        self.SegAndCropTime = None
        self.CalSegNodeDone = None
        self.CalTime = None

    def setDefaultParameters(self, parameterNode):
        """
    Initialize parameter node with default settings.
    """
        if not parameterNode.GetParameter("URL"):
            parameterNode.SetParameter("URL", "http://localhost:5000")
        if not parameterNode.GetParameter("Local"):
            parameterNode.SetParameter("Local", "true")
        if not parameterNode.GetParameter("HeartModelPath"):
            Path = RepoRoot + '/Models/Segmentation/Models_Saved/Heart_Localization'
            if os.path.exists(Path):
                parameterNode.SetParameter("HeartModelPath", Path)

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

    def SetDefaultClassVariables(self):
        self.Local = None
        self.Partial = None
        self.HeartSegNode = None
        self.HeartSeg3D = None
        self.CalSegNode = None
        self.CalSeg3D = None
        self.CroppingEnabled = None
        self.SegAndCrop = None
        self.HeartModelPath = None
        self.CalModelPath = None
        self.ServerURL = None
        self.SegAndCropDone = None
        self.HeartSegDone = None
        self.HeartSegNodeDone = None
        self.CoordinatesCalculated = None
        self.CroppingDone = None
        self.Segmentation = None
        self.SegmentationTime = None
        self.Coordinates = None
        self.VolumeIJKToRAS = None
        self.VolumeArray = None
        self.InputVolumeNode = None
        self.FinishedCallback = None
        self.UpdateCallback = None
        self.CroppingDone = None
        self.HeartSegmentationProcess = None
        self.SegAndCropTime = None
        self.CalSegNodeDone = None
        self.CalTime = None

    def SetParametersFromNode(self, InputVolumeNode, parameterNode):
        self.InputVolumeNode = InputVolumeNode
        self.parameterNode = parameterNode

    def run(self):
        print("Hi")
        self.StartOperations(self.InputVolumeNode, self.parameterNode, self.UpdateCallback, self.FinishedCallback)

    def StartOperations(self, InputVolumeNode, parameterNode, UpdateCallback=None, FinishedCallback=None):

        # Extract All Parameters From The Parameter Node
        self.SegAndCropDone = False
        self.HeartSegDone = False
        self.HeartSegNodeDone = False
        self.CoordinatesCalculated = False
        self.CroppingDone = False
        self.Local = bool(strtobool(parameterNode.GetParameter("Local")))
        self.Partial = bool(strtobool(parameterNode.GetParameter("Partial")))
        self.HeartSegNode = bool(strtobool(parameterNode.GetParameter("HeartSegNode")))
        self.HeartSeg3D = bool(strtobool(parameterNode.GetParameter("HeartSeg3D")))
        self.CalSegNode = bool(strtobool(parameterNode.GetParameter("CalSegNode")))
        self.CalSeg3D = bool(strtobool(parameterNode.GetParameter("CalSeg3D")))
        self.CroppingEnabled = bool(strtobool(parameterNode.GetParameter("CroppingEnabled")))
        self.SegAndCrop = bool(strtobool(parameterNode.GetParameter("SegAndCrop")))
        self.HeartModelPath = parameterNode.GetParameter("HeartModelPath")
        self.CalModelPath = parameterNode.GetParameter("CalModelPath")
        self.ServerURL = parameterNode.GetParameter("URL")

        # Store Volume Node
        self.InputVolumeNode = InputVolumeNode

        # Get IJKToRAS Matrix
        # Required to get some data from the volume (spacing, margin, etc.)
        # This data is used after that to show the segmentation node after finished the processing
        self.VolumeIJKToRAS = vtk.vtkMatrix4x4()
        InputVolumeNode.GetIJKToRASMatrix(self.VolumeIJKToRAS)

        # Store Callbacks
        self.FinishedCallback = FinishedCallback
        # Initialize Variables
        self.Segmentation = []
        self.SegmentationTime = 0
        self.Coordinates = []
        self.VolumeArray = np.array(slicer.util.arrayFromVolume(InputVolumeNode), copy=True)

        # Online Prerequisites Check
        if not self.Local:
            try:
                request = requests.get(self.ServerURL)
            except ConnectionError:
                print('Couldn\'t Connect To The Server')
                slicer.util.errorDisplay("Couldn't Connect To The Server")
                raise ValueError("Couldn't Connect To The Server")

        self.RunOperations()

    def RunOperations(self):

        try:
            # CLI Tests Start
            # OutputVolume = []
            # self.logic.CLITest(VolumeArray, self.LocalProcessing, ServerURL, Partial, HeartModelPath,
            #                    OutputVolume)
            # OutputVolume = self._parameterNode.GetParameter("outputVolume")
            # print(OutputVolume)
            # CLI Tests End

            # Check For Dependencies & Install Missing Ones
            if self.Local:
                self.CheckDependencies()
                self.DependenciesChecked = True

            # Compute output
            if not (self.HeartSegDone or self.SegAndCropDone):
                if self.SegAndCrop and not self.Local and not self.SegAndCropDone:

                    self.Coordinates = self.SegmentAndCrop(self.VolumeArray, self.Local,
                                                           self.ServerURL, self.HeartModelPath)
                    self.SegAndCropDone = True

                elif self.HeartSegNode or self.CroppingEnabled and not self.HeartSegDone and \
                        (self.DependenciesChecked == self.Local):
                    self.SegmentationProcessWrapper("Segmentation.slicer.py", self.VolumeArray, self.Local,
                                                    self.ServerURL, self.Partial, self.HeartModelPath,
                                                    self.HeartSegmentationCompleted)

            if self.CroppingEnabled and not self.SegAndCrop and self.HeartSegDone and not self.CoordinatesCalculated:

                self.Coordinates = self.GetCoordinates(self.Segmentation, self.Partial, 20)
                self.CoordinatesCalculated = True

            elif self.CroppingEnabled and self.SegAndCrop and not self.CoordinatesCalculated and self.SegAndCropDone:
                # Add margin to the segmentation output
                self.Coordinates = self.AddMargin(Volume=self.VolumeArray, ROICoordinates=self.Coordinates, Margin=20)
                self.CoordinatesCalculated = True

            if self.CroppingEnabled or self.SegAndCrop and self.CoordinatesCalculated and not self.CroppingDone:
                logging.info(f"The Cropping Coordinates Are Z->{self.Coordinates[0]}:{self.Coordinates[1]}, "
                             f"X->{self.Coordinates[2]}:{self.Coordinates[3]}, "
                             f"Y->{self.Coordinates[4]}:{self.Coordinates[5]}")

                NewVolume = self.CropVolume(Volume=self.VolumeArray, Coordinates=self.Coordinates)
                slicer.util.updateVolumeFromArray(self.InputVolumeNode, NewVolume)
                logging.info(f"Cropped!")
                self.CroppingDone = True

            if not self.Partial and self.HeartSegNode and not self.HeartSegNodeDone \
                    and self.HeartSegDone and (self.CroppingDone == self.CroppingEnabled):

                if self.CroppingEnabled:
                    self.Segmentation = self.CropVolume(Volume=self.Segmentation, Coordinates=self.Coordinates)
                self.CreateSegmentationNode(self.Segmentation, "Heart", self.VolumeIJKToRAS, self.HeartSeg3D)
                self.HeartSegNodeDone = True

        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: " + str(e))
            import traceback
            traceback.print_exc()

        if (self.SegAndCrop == self.SegAndCropDone) and (self.HeartSegNode == self.HeartSegNodeDone) \
                and (self.CroppingEnabled == self.CroppingDone):
            self.FinishedCallback()
            self.SetDefaultClassVariables()

    def CLITest(self, inputVolume, LocalProcessing=True, ProcessingURL="http://localhost:5000", Partial=True,
                ModelPath=None, outputVolume=None):

        import time
        startTime = time.time()
        logging.info('Processing started')

        CompressedVolume = BytesIO()
        np.savez_compressed(CompressedVolume, Volume=inputVolume)
        CompressedVolume.seek(0)
        VolumeString = CompressedVolume.read()
        VolumeStringDecoded = VolumeString.decode('UTF-8', errors="ignore")
        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'inputVolume': VolumeStringDecoded,
            'LocalProcessing': LocalProcessing,
            'ProcessingURL': ProcessingURL,
            'Partial': Partial,
            'ModelPath': ModelPath,
            'outputVolume': outputVolume
        }
        cliNode = slicer.cli.run(slicer.modules.segmentationfrommodel, None, cliParams, wait_for_completion=True)

        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info('Processing completed in {0:.2f} seconds'.format(stopTime - startTime))

    def SegmentAndCrop(self, inputVolume, Local=True, ProcessingURL="http://localhost:5000",
                       ModelPath=None):

        # TODO: Receive Routes From Caller
        if inputVolume is None:
            raise ValueError("Input volume is invalid")

        startTime = time.time()
        logging.info('Processing started')

        # Convert Volume To NumPy Array
        VolumeArray = np.copy(inputVolume)

        # Prepare Slices
        RawSliceArrays, files, ShiftValues = self.GetSampleSlicesFromVolume(VolumeArray=VolumeArray,
                                                                            Local=Local)

        # Send to Server For Processing
        if not Local:
            SliceSendReq = requests.post(ProcessingURL + "/crop", files=files, data=ShiftValues)
            Coordinates = SliceSendReq.json()["Coor"]
            logging.info(f"Received Cropping Coordinates From Online Server")
        else:
            from Models.Segmentation.Inference import Infer
            model = Infer(model_path=ModelPath, model_input=(112, 112, 112))
            AxSeg = model.predict(np.array(RawSliceArrays[0]))
            SagSeg = model.predict(np.array(RawSliceArrays[1]))
            CorSeg = model.predict(np.array(RawSliceArrays[2]))

            AxCoor = [int(i) for i in get_coords(AxSeg)]
            SagCoor = [int(i) for i in get_coords(SagSeg)]
            CorCoor = [int(i) for i in get_coords(CorSeg)]

            Coordinates = [AxCoor, SagCoor, CorCoor]

            logging.info(f"Cropping Coordinates Calculated Locally")

        logging.info(f"The Cropping Coordinates Are {Coordinates}")
        stopTime = time.time()
        logging.info(
            'Segmentation & Coordinates Calculation Completed in in {0:.2f} seconds'.format(stopTime - startTime))

        return Coordinates

    def Segment(self, inputVolume, LocalProcessing=True, ProcessingURL="http://localhost:5000", Partial=True,
                ReturnTime=True, ModelPath=None):

        # TODO: Receive Routes From Caller
        if inputVolume is None:
            raise ValueError("Input volume is invalid")

        if ModelPath is None:
            raise ValueError("Model Path Is Incorrect")

        # Get Segmentation Start Time
        SegmentStart = time.time()

        # Convert Volume To NumPy Array
        VolumeArray = np.copy(inputVolume)
        VolumeShape = VolumeArray.shape
        SegmentedSlices = []

        # Segment 3 Slicers From Each View
        if Partial:
            SegmentedSlices = [[], [], []]
            RawSliceArrays, files, ShiftValues = self.GetSampleSlicesFromVolume(VolumeArray=VolumeArray,
                                                                                Local=LocalProcessing)

            if not LocalProcessing:
                # Send Data To Server For Processing
                SliceSendReq = requests.post(ProcessingURL + "/segment/slices", files=files, data=ShiftValues)
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

        if ReturnTime:
            return SegmentedSlices, SegmentTime
        else:
            return SegmentedSlices

    def SegmentationProcessWrapper(self, ScriptName, VolumeArray, Local, ServerURL, Partial, HeartModelPath,
                                   CompletedCallback):

        scriptFolder = slicer.modules.cascoremodule.path.replace('CaScoreModule.py', '/Resources/ProcessScripts/')
        scriptPath = os.path.join(scriptFolder, ScriptName)
        self.HeartSegmentationProcess = SegmentationProcess(scriptPath, VolumeArray, Local,
                                                            ServerURL, Partial,
                                                            HeartModelPath)
        logic = ProcessesLogic(completedCallback=lambda: CompletedCallback())
        logic.addProcess(self.HeartSegmentationProcess)
        logic.run()
        logging.info('Segmentation Process Started')

    def HeartSegmentationCompleted(self):
        """
        Callback Function Called When Heart Segmentation Process is Completed, Sets Completion Variables
        And Extracts Data
        """
        self.HeartSegDone = True
        self.Segmentation = self.HeartSegmentationProcess.Output["Segmentation"]
        self.SegmentationTime = self.HeartSegmentationProcess.Output["SegmentationTime"]
        logging.info('Segmentation completed in {0:.2f} seconds'.format(self.SegmentationTime))
        self.RunOperations()

    def CheckDependencies(self):
        """
        Installs Missing Pip Packages
        """
        # Install Dependencies if Not Detected
        Scikit = importlib.util.find_spec("scikit-image")
        TensorFlow = importlib.util.find_spec("tensorflow")

        if Scikit is None:
            logging.info('Installing Scikit')
            pip_install("scikit-image")
            logging.info('Scikit Installed')
        else:
            logging.info('Scikit Found')

        if TensorFlow is None:
            logging.info('Installing TensorFlow')
            pip_install("tensorflow")
            logging.info('TensorFlow Installed')
        else:
            logging.info('TensorFlow Found')

    def CreateSegmentationNode(self, Segmentation, Name=None, VolumeIJKToRAS=None, HeartSegVis=False):
        """
        Creates A Segmentation Node From The Given Segmentation
        :param Segmentation: NumPy Array Containing The Segmentation
        :param Name: Name TO Give To The Segmentation Node
        :param VolumeIJKToRAS: IJKToRAS Matrix of The Original Volume
        :param HeartSegVis: If True Creates a Closed Surface Representation of The Segmentation (3D View)
        """
        # Create a new LabelMapVolume
        LabelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', f'{Name}-Label')

        # Set IJKToRAS Matrix To That of The Original Volume
        if VolumeIJKToRAS:
            LabelMapVolumeNode.SetIJKToRASMatrix(VolumeIJKToRAS)

        # Copy Segmentation
        SegmentationNp = np.copy(Segmentation)

        # Update the LabelMapVolume from the given Segmentation array
        slicer.util.updateVolumeFromArray(LabelMapVolumeNode, SegmentationNp)

        # Create a SegmentationNode
        SegNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f'{Name}-Segmentation')

        # Load the LabelMapVolume into the SegmentationNode
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(LabelMapVolumeNode, SegNode)

        if HeartSegVis:
            # Create Closed Surface Representation
            SegNode.CreateClosedSurfaceRepresentation()

        # Delete The LabelMapVolume
        slicer.mrmlScene.RemoveNode(LabelMapVolumeNode)

    def GetCoordinates(self, Segmentation, Partial, Margin):
        """
        Gets The Cropping Parameters of The Given Segmentation
        :param Segmentation: Segmentation Showing The Area of Interest
        :param Partial: Whether The Given Segmentation is Partial (Slices in Each View) or The Full Volume
        :param Margin: Margin Value Added To Calculated Coordinates
        :return Coordinates: List Containing The Cropping Coordinates in Each Direction [z1, z2, x1, x2, y1, y2]
        """
        # Get CroppingCoordinates
        ROICoordinates = GetCoords(Segmentation, Partial)

        # Add Margin
        Coordinates = self.AddMargin(Segmentation, ROICoordinates, Margin)

        return Coordinates

    def AddMargin(self, Volume, ROICoordinates, Margin):
        """
        Gets The Cropping Parameters of The Given Segmentation
        :param Volume: Original Volume/Segmentation
        :param ROICoordinates: ROI Coordinates
        :param Margin: Margin Value Added To Calculated Coordinates
        :return Coordinates: List Containing The Cropping Coordinates in Each Direction [z1, z2, x1, x2, y1, y2]
        """

        # Add Margin
        z1 = (ROICoordinates[0] - Margin) if (ROICoordinates[0] - Margin >= 0) else 0
        z2 = (ROICoordinates[1] + Margin) if (ROICoordinates[1] + Margin < Volume.shape[0]) else Volume.shape[0]
        x1 = (ROICoordinates[2] - Margin) if (ROICoordinates[2] - Margin >= 0) else 0
        x2 = (ROICoordinates[3] + Margin) if (ROICoordinates[3] + Margin < Volume.shape[1]) else Volume.shape[1]
        y1 = (ROICoordinates[4] - Margin) if (ROICoordinates[4] - Margin >= 0) else 0
        y2 = (ROICoordinates[5] + Margin) if (ROICoordinates[5] + Margin < Volume.shape[2]) else Volume.shape[2]

        Coordinates = [z1, z2, x1, x2, y1, y2]

        return Coordinates

    def CropVolume(self, Volume, Coordinates):
        """
        Crops The Volume Using The Given Coordinates
       :param Volume: Volume To Be Cropped
       :param Coordinates: List Containing The Cropping Cordinates in Each Direction [z1, z2, x1, x2, y1, y2]
       :return CroppedVolume: ndarray of the cropped volume
       """
        CroppedVolume = np.copy(Volume[Coordinates[0]: Coordinates[1] + 1, Coordinates[2]: Coordinates[3] + 1,
                                Coordinates[4]: Coordinates[5] + 1])
        return CroppedVolume

    def CropVolumeWithSegmentation(self, Volume, Segmentation, Partial, Margin):
        """
       Crops The Volume Using The Area of Interest Found in The Segmentation
        :param Volume: Volume To Be Cropped
       :param Segmentation: Segmentation Showing The Area of Interest
       :param Partial: Whether The Given Segmentation is Partial (Slices in Each View) or The Full Volume
       :param Margin: Margin Value Added To Calculated Coordinates
       :return CroppedVolume: NumPy Array of The Cropped Volume
       """
        Coordinates = self.GetCoordinates(Segmentation, Partial, Margin)
        CroppedVolume = self.CropVolume(Volume, Coordinates)
        return CroppedVolume

    def GetSampleSlicesFromVolume(self, VolumeArray=None, Local=True):

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
