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
    from pymedphys import gamma
except:
    from slicer.util import pip_install
    #pip_install('pymedphys --no-deps')
    pip_install('pymedphys')
    from pymedphys import gamma

try:
  import skimage.metrics as compare
except:
    from slicer.util import pip_install
    pip_install('scikit-image')
    import skimage.metrics as compare

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

    def runTest(self, msec=100, **kwargs):
        """
        :param msec: delay to associate with :func:`ScriptedLoadableModuleTest.delayDisplay()`.
        """
        # test GammaTest
        # name of the test case class is expected to be <ModuleName>Test
        module = importlib.import_module(self.__module__)
        className = self.moduleName + 'Test'
        try:
            TestCaseClass = getattr(module, className)
        except AttributeError:
            # Treat missing test case class as a failure; provide useful error message
            raise AssertionError(
                f'Test case class not found: {self.__module__}.{className} ')

        testCase = TestCaseClass()
        testCase.messageDelay = msec
        testCase.runTest(**kwargs)

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

        # distance_threshold=1, interp_fraction=10, dose_threshold=1, lower_dose_cutoff=20
        self.distanceThreshold = qt.QDoubleSpinBox()
        self.distanceThreshold.setDecimals(1)
        self.distanceThreshold.setMinimum(0)
        self.distanceThreshold.setMaximum(10)
        self.distanceThreshold.value = 1
        self.distanceThreshold.setToolTip("This is the distance (mm) to do gamma comparisons")

        self.interpolationFactor = qt.QSpinBox()
        self.interpolationFactor.setMinimum(1)
        self.interpolationFactor.setMaximum(30)
        self.interpolationFactor.value=10
        self.interpolationFactor.setToolTip("This is the factor to divide the grid for interpolations")

        self.doseThreshold = qt.QDoubleSpinBox()
        self.doseThreshold.setDecimals(1)
        self.doseThreshold.setMinimum(0.1)
        self.doseThreshold.setMaximum(10)
        self.doseThreshold.value = 1
        self.doseThreshold.setToolTip("This is the dose threshold (%) to accept the gamma comparison")

        self.lowerDoseCutoff = qt.QDoubleSpinBox()
        self.lowerDoseCutoff.setDecimals(1)
        self.lowerDoseCutoff.setMinimum(0.1)
        self.lowerDoseCutoff.setMaximum(10)
        self.lowerDoseCutoff.value = 1
        self.lowerDoseCutoff.setToolTip("This is the dose cutoff (%) to reject dose values")

        ParametersFormLayout.addRow("Distance Threshold (mm)", self.distanceThreshold)
        ParametersFormLayout.addRow("Interpolation Factor", self.interpolationFactor)
        ParametersFormLayout.addRow("Dose Threshold (%)", self.doseThreshold)
        ParametersFormLayout.addRow("Lower Dose Cutoff (%)", self.lowerDoseCutoff)

        InputCollapsibleButton = ctk.ctkCollapsibleButton()
        InputCollapsibleButton.text = "Input Volumes"
        self.layout.addWidget(InputCollapsibleButton)
        InputLayout = qt.QHBoxLayout(InputCollapsibleButton)
        InputWidget = qt.QWidget()
        InputLayout.addWidget(InputWidget)
        InputFormLayout = qt.QFormLayout(InputWidget)

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
        InputFormLayout.addRow(
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
        InputFormLayout.addRow(
            "ADR Image 2: ", self.Image2)

        self.ApplyButton = qt.QPushButton("Calculate Gamma")
        self.ApplyButton.toolTip = "Calculates Gamma image."
        InputFormLayout.addRow(self.ApplyButton)

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
        distance_threshold = self.distanceThreshold.value
        interp_fraction = self.interpolationFactor.value
        dose_threshold = self.doseThreshold.value
        lower_dose_cutoff = self.lowerDoseCutoff.value
        self.logic.run(
            inputVolume1=self.Image1.currentNode(), 
            inputVolume2=self.Image2.currentNode(),
            distance_threshold=distance_threshold, 
            interp_fraction=interp_fraction, 
            dose_threshold=dose_threshold, 
            lower_dose_cutoff=lower_dose_cutoff
        )


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

    def showTable(self, table):
        """
        Switch to a layout where tables are visible and show the selected table
        """
        layoutManager = slicer.app.layoutManager()
        currentLayout = layoutManager.layout
        layoutWithTable = slicer.modules.tables.logic().GetLayoutWithTable(currentLayout)
        layoutManager.setLayout(layoutWithTable)
        appLogic = slicer.app.applicationLogic()
        appLogic.GetSelectionNode().SetActiveTableID(table.GetID())
        appLogic.PropagateTableSelection()


    def skimage_metrics(self, im1, im2):
        ## Metric 1: MSE (Mean Squared Error)
        mse = compare.mean_squared_error(im1, im2)
        ## Metric 1.1: NRMSE (Normalized Root Mean Squared Error)
        nrmse = compare.normalized_root_mse(im1, im2)
        ## Metric 2: SSIM (Structural Similarity Image Matrix)
        ssim = compare.structural_similarity(im1, im2, data_range= im2.max() - im2.min())
        ## Metric 3: Peak Signal-to-Noise ratio
        psnr = compare.peak_signal_noise_ratio(im1, im2, data_range= im2.max() - im2.min())
        return mse, nrmse, ssim, psnr

    def GammaIndex(self, im1, im2, imageThreshold, distance_threshold, interp_fraction, dose_threshold, lower_dose_cutoff):
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
        gamma_options = {
            'dose_percent_threshold': dose_threshold,
            'distance_mm_threshold': distance_threshold,
            'lower_percent_dose_cutoff': lower_dose_cutoff,
            'interp_fraction': interp_fraction,  # Should be 10 or more for more accurate results
            'max_gamma': 2,
            'random_subset': None,
            'local_gamma': False,
            'ram_available': 2**29  # 1/2 GB
        }
        
        gamma_const = gamma(coords1, im1, coords2, im2, **gamma_options)
        valid_gamma_const = np.ma.masked_invalid(gamma_const)
        valid_gamma_const = valid_gamma_const[~valid_gamma_const.mask]
        size = len(valid_gamma_const)
        if size<1:
            size=1
        gamma_index = np.sum(valid_gamma_const <= 1) / size
        return gamma_index, gamma_const

    def run(self, inputVolume1, inputVolume2, distance_threshold=1, interp_fraction=10, dose_threshold=1, lower_dose_cutoff=20):
        im1 = slicer.util.arrayFromVolume(inputVolume1).astype(float)
        im2 = slicer.util.arrayFromVolume(inputVolume2).astype(float)
        GammaImage = self.cloneNode(inputVolume1, "Gamma Image")
        imageThreshold = inputVolume1.GetSpacing()
        GammaIndex, GammaMatrix = self.GammaIndex(im1, im2, imageThreshold, distance_threshold, interp_fraction, dose_threshold, lower_dose_cutoff)
        slicer.util.updateVolumeFromArray(GammaImage, GammaMatrix)
        mse_const, nrmse_const, ssim_const, psnr_const = self.skimage_metrics(im1, im2)
        TableValues={
            "Gamma Index": GammaIndex, 
            "Mean Square Error (MSE)": mse_const, 
            "Normalized Root MSE (%)": nrmse_const, 
            "Structural Similarity Index": ssim_const, 
            "Peak Signal to Noise Ratio": psnr_const
        }
        displayNode = GammaImage.GetScalarVolumeDisplayNode()

        if displayNode is not None:
            colorID = slicer.util.getFirstNodeByName("DivergingBlueRed").GetID()
            displayNode.SetAndObserveColorNodeID(colorID)
            displayNode.AutoWindowLevelOff()
            displayNode.AutoWindowLevelOn()

        TableNodes = slicer.util.getNodesByClass('vtkMRMLTableNode')
        name = f'T:TABL Comparison'
        for node in TableNodes:
            if name == node.GetName():  # Table exists, erase it
                slicer.mrmlScene.RemoveNode(node)
        
        # prepare clean table
        resultsTableNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLTableNode')
        resultsTableNode.RemoveAllColumns()
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        nodeID = shNode.GetItemByDataNode(inputVolume1)
        folderID = shNode.GetItemParent(nodeID)
        shNode.CreateItem(folderID, resultsTableNode)
        resultsTableNode.SetName(name)
        table = resultsTableNode.GetTable()

        segmentColumnValue = vtk.vtkStringArray()
        segmentColumnValue.SetName("Metric")
        table.AddColumn(segmentColumnValue)

        segmentColumnValue = vtk.vtkStringArray()
        segmentColumnValue.SetName("Value")
        table.AddColumn(segmentColumnValue)

        table.SetNumberOfRows(len(TableValues.keys()))

        for i, (metric, value) in enumerate(TableValues.items()):
            table.SetValue(i, 0, metric)
            table.SetValue(i, 1, value)

        resultsTableNode.Modified()
        self.showTable(resultsTableNode)

        backgroundID = inputVolume1.GetID()
        foregroundID = GammaImage.GetID()

        slicer.util.setSliceViewerLayers(
            background=str(backgroundID), 
            foreground=str(foregroundID),   
            foregroundOpacity=0.5
        )

        return TableValues  # for testing

#
# GammaTest
#

def hasImageData(volumeNode):
    """
    This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
        logger.info(f"{utils.whoami()} failed: no volume node")
        return False
    if volumeNode.GetImageData() is None:
        logger.info(f"{utils.whoami()} failed: no image data in volume node")
        return False
    return True


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
        slicer.app.processEvents()

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
        slicer.app.aboutToQuit.connect(self.myCleanup)

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

        # Test the module logic
        logic = GammaLogic()

        # Test algorithm 
        values = logic.run(inputVolume, inputVolume, distance_threshold=1, interp_fraction=10, dose_threshold=1, lower_dose_cutoff=1)
        OutputNode = slicer.util.getFirstNodeByClassByName(
            'vtkMRMLScalarVolumeNode', 'Gamma Image')
        logging.info(str(values))

        self.assertEqual(values["Gamma Index"], 1)
        self.assertEqual(values["Normalized Root MSE (%)"], 0)
        self.assertEqual(values["Structural Similarity Index"], 1)
        self.assertTrue(hasImageData(OutputNode))

        self.delayDisplay('Test passed')

