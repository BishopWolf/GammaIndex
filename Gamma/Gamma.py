import os
import unittest
import logging
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *

import numpy as np

try:
    # from pymedphys.gamma import gamma_shell as calc_gamma
    from pymedphys._gamma.implementation import gamma_shell
except:
    from slicer.util import pip_install
    pip_install('pymedphys --no-deps')
    from pymedphys._gamma.implementation import gamma_shell

try:
    from npgamma import calc_gamma
except:
    from slicer.util import pip_install
    pip_install('npgamma')
    from npgamma import calc_gamma

#
# Gamma
#


class Gamma(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Gamma"  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Radiotherapy"]
        # TODO: add here list of module names that this module requires
        self.parent.dependencies = []
        # TODO: replace with "Firstname Lastname (Organization)"
        self.parent.contributors = ["Alex Vergara Gil (INSERM, France)", "Gan Quan (INSERM, France)"]
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
"""  # TODO: update with short description of the module
        self.parent.helpText += self.getDefaultModuleDocumentationLink(
        )  # TODO: verify that the default URL is correct or change it to the actual documentation
        self.parent.acknowledgementText = """

"""  # TODO: replace with organization, grant and thanks.

#
# GammaWidget
#


class GammaWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        super().setup()

        ParametersCollapsibleButton = ctk.ctkCollapsibleButton()
        ParametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(ParametersCollapsibleButton)
        ParameterLayout = qt.QHBoxLayout(ParametersCollapsibleButton)
        ParametersWidget = qt.QWidget()
        ParameterLayout.addWidget(ParametersWidget)
        ParametersFormLayout = qt.QFormLayout(ParametersWidget)

        # input volume selector
        self.Image1 = slicer.qMRMLNodeComboBox()
        self.Image1.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.Image1.selectNodeUponCreation = True
        self.Image1.addEnabled = False
        self.Image1.removeEnabled = False
        self.Image1.noneEnabled = False
        self.Image1.showHidden = False
        self.Image1.showChildNodeTypes = False
        self.Image1.setMRMLScene(slicer.mrmlScene)
        self.Image1.setToolTip("Select ADR Image 1.")
        ParametersFormLayout.addRow(
            "ADR Image 1: ", self.Image1)

        # input volume selector
        self.Image2 = slicer.qMRMLNodeComboBox()
        self.Image2.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.Image2.selectNodeUponCreation = True
        self.Image2.addEnabled = False
        self.Image2.removeEnabled = False
        self.Image2.noneEnabled = False
        self.Image2.showHidden = False
        self.Image2.showChildNodeTypes = False
        self.Image2.setMRMLScene(slicer.mrmlScene)
        self.Image2.setToolTip("Select ADR Image 1.")
        ParametersFormLayout.addRow(
            "ADR Image 2: ", self.Image2)

        # output volume selector
        self.Image3 = slicer.qMRMLNodeComboBox()
        self.Image3.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.Image3.selectNodeUponCreation = True
        self.Image3.addEnabled = True
        self.Image3.removeEnabled = True
        self.Image3.noneEnabled = True
        self.Image3.showHidden = False
        self.Image3.showChildNodeTypes = False
        self.Image3.setMRMLScene(slicer.mrmlScene)
        self.Image3.setToolTip("Gamma Image.")
        ParametersFormLayout.addRow(
            "Gamma Image: ", self.Image3)


        self.ApplyButton = qt.QPushButton("Calculate Gamma")
        self.ApplyButton.toolTip = "Calculates Gamma image."
        ParametersFormLayout.addRow(self.ApplyButton)

        self.ApplyButton.clicked.connect(self.onApplyButton)

         # Add vertical spacer
        self.layout.addStretch(1)

        # Create a new parameterNode
        # This parameterNode stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.logic = GammaLogic()


    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        self.logic.run(self.Image1.currentNode(), self.Image2.currentNode(), self.Image3.currentNode())


#
# GammaLogic
#

class GammaLogic(ScriptedLoadableModuleLogic):
    def cloneNode(self, node, newName):
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        nodeID = shNode.GetItemByDataNode(node)
        newnode = slicer.util.getFirstNodeByClassByName(
            'vtkMRMLScalarVolumeNode', newName)

        if newnode:
            slicer.mrmlScene.RemoveNode(newnode)

        clonedNode = slicer.modules.volumes.logic().CloneVolume(node, newName)
        clonedItemID = shNode.GetItemByDataNode(clonedNode)
        clonedNodeName = clonedNode.GetName()
        if clonedNodeName != newName:
            shNode.SetItemName(clonedItemID, newName)

        parent = shNode.GetItemParent(nodeID)
        shNode.SetItemParent(clonedItemID, parent)

        transformNode = node.GetParentTransformNode()
        if transformNode is not None:
            clonedNode.SetAndObserveTransformNodeID(transformNode.GetID())

        return clonedNode

    def GammaIndex(self, im1, im2, imageThreshold, distance_step_size, dose_threshold, lower_dose_cutoff):
        shape = np.shape(im1)
        xgrid = np.linspace(1,shape[0]*imageThreshold[0],shape[0])
        ygrid = np.linspace(1,shape[1]*imageThreshold[1],shape[1])
        zgrid = np.linspace(1,shape[2]*imageThreshold[2],shape[2])
        coords1 = (xgrid, ygrid, zgrid)
        shape = np.shape(im2)
        xgrid = np.linspace(1,shape[0]*imageThreshold[0],shape[0])
        ygrid = np.linspace(1,shape[1]*imageThreshold[1],shape[1])
        zgrid = np.linspace(1,shape[2]*imageThreshold[2],shape[2])
        coords2 = (xgrid, ygrid, zgrid)
        ## Gamma index parameters
        distance_threshold = imageThreshold[0]
        
        gamma_const = gamma_shell(coords1, im1, coords2, im2,
            distance_mm_threshold=distance_threshold, dose_percent_threshold=dose_threshold,
            lower_percent_dose_cutoff=lower_dose_cutoff, 
            interp_fraction=distance_step_size,
            max_gamma=10, local_gamma=True)
        valid_gamma_const = np.ma.masked_invalid(gamma_const)
        valid_gamma_const = valid_gamma_const[~valid_gamma_const.mask]
    #     valid_gamma_const[valid_gamma_const > 2] = 2
        size = len(valid_gamma_const)
        if size<1:
            size=1
        return np.sum(valid_gamma_const <= 1) / size, gamma_const

    def GammaIndex2(self, im1, im2, imageThreshold, distance_step_size, dose_threshold, lower_dose_cutoff):
        shape = np.shape(im1)
        xcoord = np.arange(0.0, shape[0]*imageThreshold[0], imageThreshold[0])
        ycoord = np.arange(0.0, shape[1]*imageThreshold[1], imageThreshold[1])
        zcoord = np.arange(0.0, shape[2]*imageThreshold[2], imageThreshold[2]) 
        coords = (xcoord, ycoord, zcoord)
        distance_threshold = imageThreshold[0]
        distance_step_size = distance_threshold / distance_step_size
        max_concurrent_calc_points = 30000000
        num_threads = 4
        dose_threshold = dose_threshold * np.max(im1)

        gamma_const = calc_gamma(coords, im1, coords, im2,
                  distance_threshold, dose_threshold,
                  lower_dose_cutoff=lower_dose_cutoff, 
                  distance_step_size=distance_step_size,
                  maximum_test_distance=np.inf,
                  max_concurrent_calc_points=max_concurrent_calc_points,
                  num_threads=num_threads)

        valid_gamma_const = np.ma.masked_invalid(gamma_const)
        valid_gamma_const = valid_gamma_const[~valid_gamma_const.mask]
    #     valid_gamma_const[valid_gamma_const > 2] = 2
        size = len(valid_gamma_const)
        if size<1:
            size=1
        return np.sum(valid_gamma_const <= 1) / size, gamma_const


    def run(self, inputVolume1, inputVolume2, outputVolume, distance_step_size=10, dose_threshold=0.02, lower_dose_cutoff=0.2):
        im1 = slicer.util.arrayFromVolume(inputVolume1).astype(float)
        im2 = slicer.util.arrayFromVolume(inputVolume2).astype(float)
        GammaImage = self.cloneNode(inputVolume1, outputVolume.GetName())
        imageThreshold = inputVolume1.GetSpacing()
        GammaIndex, GammaMatrix = self.GammaIndex(im1, im2, imageThreshold, distance_step_size, dose_threshold, lower_dose_cutoff)
        slicer.util.updateVolumeFromArray(GammaImage, GammaMatrix)



#
# GammaTest
#


class GammaTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_Gamma1()

    def test_Gamma1(self):
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
        inputVolume = SampleData.downloadFromURL(
            nodeNames='MRHead',
            fileNames='MR-Head.nrrd',
            uris='https://github.com/Slicer/SlicerTestingData/releases/download/MD5/39b01631b7b38232a220007230624c8e',
            checksums='MD5:39b01631b7b38232a220007230624c8e')[0]
        self.delayDisplay('Finished with download and loading')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 279)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode")
        threshold = 50

        # Test the module logic

        logic = GammaLogic()

        # Test algorithm with non-inverted threshold
        logic.run(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.run(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
