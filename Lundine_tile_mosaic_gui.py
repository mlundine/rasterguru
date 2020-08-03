from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import sys
import glob
import gdal_retile
import gdal_merge
import numpy
import six


## Contains all of the widgets for the GUI   
class Window(QMainWindow):
    
    ## All of the button actions are functions
    ## Initializing the window
    def __init__(self):
        super(Window, self).__init__()
        sizeObject = QDesktopWidget().screenGeometry(-1)
        global screenWidth
        screenWidth = sizeObject.width()
        global screenHeight
        screenHeight = sizeObject.height()
        self.setGeometry(50, 50, 10 + int(screenWidth/2), 10 + int(screenHeight/2))
        self.setWindowTitle("Raster Guru")
        self.home()
    def home(self):
        mosaic = QPushButton('Make Mosaic', self)
        mosaic.resize(200,100)
        mosaic.move(300,100)
        resolution = QSpinBox(self)
        resolution.setMinimum(1)
        resolution.setMaximum(500)
        resolution.setValue(1)
        resolution.move(300,220)
        resolutionLab = QLabel('Resolution (pixels)', self)
        resolutionLab.resize(200,25)
        resolutionLab.move(300,200)
        
        tile = QPushButton('Make Tiles', self)
        tile.resize(200,100)
        tile.move(100,100)
        heightWidth = QSpinBox(self)
        heightWidth.setMinimum(10)
        heightWidth.setMaximum(5000)
        heightWidth.setValue(300)
        heightWidth.move(100,220)
        heightWidthLab = QLabel('Height and Width (pixels)', self)
        heightWidthLab.resize(200,25)
        heightWidthLab.move(100,200)
        overlap = QSpinBox(self)
        overlap.move(100,275)
        overlap.setMinimum(0)
        overlap.setMaximum(5000)
        overlapLab = QLabel('Overlap (pixels)', self)
        overlapLab.move(100,250)
        overlapLab.resize(200,25)
        #actions
        tile.clicked.connect(lambda: self.tileButton(heightWidth.value(), overlap.value()))
        mosaic.clicked.connect(lambda: self.mosaicButton(resolution.value()))
        self.show()
    def tileButton(self, dim, overlap):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select raster to tile", "","All Files (*);;Geotiffs (*.tif)", options=options)
        if fileName:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            folderName = str(QFileDialog.getExistingDirectory(self, "Select Folder for Saving"))
            if folderName:
                tiler = gdal_retile
                tiler.Verbose = True
                tiler.TileHeight = dim
                tiler.TileWidth = dim
                tiler.Overlap = overlap
                tiler.Names = [fileName]
                tiler.TargetDir = folderName + r'/'
                tiler.TileIndexName = os.path.splitext(os.path.basename(fileName))[0] + '_tiles_shapefile'
                tiler.main()
                
                #cmd1 = 'python gdal_retile.py -ps ' + str(dim) + ' ' + str(dim) + ' -targetDir ' + folderName
                cdm1 = '-ps ' + str(dim) + ' ' + str(dim) + ' -targetDir ' + folderName
                cmd2 = ' -overlap ' + str(overlap) + ' ' + '-r near -tileIndex ' + os.path.splitext(os.path.basename(fileName))[0] + '_tiles'
                cmd3 = ' -csv ' + os.path.splitext(os.path.basename(fileName))[0] + '_tiles' + ' -csvDelim , '+ fileName
                #os.system(cmd1+cmd2+cmd3)
    def mosaicButton(self, res):
        def listToString(s):      
            # initialize an empty string 
            str1 = " " 
            # return string
            return (str1.join(s)) 
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = str(QFileDialog.getExistingDirectory(self, "Select Folder of Rasters"))
        if folderName:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName,_ = QFileDialog.getSaveFileName(self, 'Save File')
            if fileName:
        
                types = ['/*.tiff', '/*.tif', '/*.img']
                files = []
                for ext in types:
                    for raster in glob.glob(folderName + ext):
                        files.append(raster)
                merger = gdal_merge
                merger.names = files
                merger.out_file = fileName+'.tif'
                merger.psize_x = float(res)
                merger.psize_y = float(-1*res)
                merger.main()
                #cmd1 = 'python gdal_merge.py -o ' + fileName + '.tif' + ' -ps ' + str(res) + ' ' + str(res) + ' ' + filestring
                #cmd1 = ['-o', fileName+'.tif', '-ps', str(res), str(res), filestring]
                #gdal_merge.main(cmd1)
                #os.system(cmd1)
                
## Function outside of the class to run the app   
def run():
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

## Calling run to run the app
run()
