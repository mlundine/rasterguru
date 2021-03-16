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
import gdal_functions_app


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
        global bw1
        bw1 = int(screenWidth/15)
        global bw2
        bw2 = int(screenWidth/50)
        global bh1
        bh1 = int(screenHeight/15)
        global bh2
        bh2 = int(screenHeight/20)
        self.setGeometry(50, 50, 10 + int(screenWidth/2), 10 + int(screenHeight/2))
        self.setWindowTitle("Raster Guru")
        self.home()
    def home(self):

        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QGridLayout()             # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
        self.widget.setLayout(self.vbox)
        
        mosaic = QPushButton('Make Mosaic')
        self.vbox.addWidget(mosaic, 3, 0)

        resolutionLab = QLabel('Resolution (pixels)')
        self.vbox.addWidget(resolutionLab, 2, 1)
        
        resolution = QSpinBox()
        resolution.setMinimum(1)
        resolution.setMaximum(500)
        resolution.setValue(1)
        self.vbox.addWidget(resolution, 3, 1)
        

        
        tile = QPushButton('Make Tiles')
        tile.move(100,100)
        self.vbox.addWidget(tile, 0, 0)

        heightWidthLab = QLabel('Height and Width (pixels)')
        self.vbox.addWidget(heightWidthLab, 0, 1)
        
        heightWidth = QSpinBox()
        heightWidth.setMinimum(10)
        heightWidth.setMaximum(5000)
        heightWidth.setValue(300)
        self.vbox.addWidget(heightWidth, 1, 1)

        overlapLab = QLabel('Overlap (pixels)')
        self.vbox.addWidget(overlapLab, 0, 2)
        
        overlap = QSpinBox()
        overlap.setMinimum(0)
        overlap.setMaximum(5000)
        self.vbox.addWidget(overlap, 1, 2)

        deleter = QPushButton('Delete Null Tiles')
        self.vbox.addWidget(deleter, 0, 3)

        converter = QPushButton('Convert Rasters')
        self.vbox.addWidget(converter, 4, 0)
        fromType = QComboBox()
        toType = QComboBox()
        fromTypes = ['.tif', '.img']
        toTypes = ['.tif', '.jpeg', '.img', '.png', '.npy']
        for t in fromTypes:
            fromType.addItem(t)
        for t in toTypes:
            toType.addItem(t)
        toLab = QLabel('To')
        
        self.vbox.addWidget(fromType, 4, 1)
        self.vbox.addWidget(toLab, 4, 2)
        self.vbox.addWidget(toType, 4, 3)

        multiband = QPushButton('Make Multiband Rasters')
        self.vbox.addWidget(multiband, 6,0)
        
        #todo
        numBandsLab = QLabel('Bands')
        self.vbox.addWidget(numBandsLab, 5, 1)
        numBands = QSpinBox()
        numBands.setValue(3)
        numBands.setMinimum(2)
        numBands.setMaximum(15)
        self.vbox.addWidget(numBands, 6, 1)

        slope = QPushButton('Slope')
        self.vbox.addWidget(slope, 7, 0)

        aspect = QPushButton('Aspect')
        self.vbox.addWidget(aspect, 8, 0)

        hillshade = QPushButton('Hillshade')
        self.vbox.addWidget(hillshade, 9, 0)

        roughness = QPushButton('Roughness')
        self.vbox.addWidget(roughness, 10, 0)

        subtract = QPushButton('Raster Difference')
        self.vbox.addWidget(subtract, 11, 0)

        firstLabel = QLabel('Full Filepath')
        first = QLineEdit()

        secondLabel = QLabel('Full Filepath')
        second = QLineEdit()

        minus = QLabel('minus')

        maskLabel = QLabel('Full Filepath to Shapefile Mask (optional)')
        mask = QLineEdit()

        densityLabel = QLabel('Density of Material')
        density = QLineEdit()
        
        self.vbox.addWidget(firstLabel, 11, 1)
        self.vbox.addWidget(first,12,1)
        self.vbox.addWidget(minus,12,2)
        self.vbox.addWidget(secondLabel,11,3)
        self.vbox.addWidget(second,12,3)
        self.vbox.addWidget(maskLabel, 11,4)
        self.vbox.addWidget(mask, 12, 4)
##        self.vbox.addWidget(densityLabel,11,5)
##        self.vbox.addWidget(density,12,5)

        kmeans = QPushButton('K-means unsupervised classification')
        self.vbox.addWidget(kmeans, 13, 0)
        kmeanClassesLabel = QLabel('# of Classes')
        kmeanClasses = QSpinBox()
        kmeanClasses.setMinimum(2)
        kmeanClasses.setValue(2)
        kmeansNoDataLabel = QLabel('NoData Value')
        kmeansNoData = QLineEdit()
        self.vbox.addWidget(kmeansNoDataLabel, 13, 2)
        self.vbox.addWidget(kmeanClassesLabel,13,1)
        self.vbox.addWidget(kmeansNoData, 14, 2)
        self.vbox.addWidget(kmeanClasses, 14, 1)
        
        #todo
        virtualRaster = QPushButton('Make Virtual Raster Dataset')
        self.vbox.addWidget(virtualRaster, 15, 0)
        
        clipRaster = QPushButton('Clip Raster to Shape')
        self.vbox.addWidget(clipRaster, 16, 0)

        rasterToShape = QPushButton('Convert Rasters To Shapefiles')
        self.vbox.addWidget(rasterToShape, 17, 0)

        mergeShapefiles = QPushButton('Merge Shapefiles')
        shapeLabel = QLabel('Full Filepath to Shapefile')
        shape = QLineEdit()
        self.vbox.addWidget(mergeShapefiles, 18, 0)
        self.vbox.addWidget(shapeLabel,18,1)
        self.vbox.addWidget(shape,19,1)

        zonalStats = QPushButton('Zonal Statistics')
        self.vbox.addWidget(zonalStats,20,0)

        saveCoordsAndRes = QPushButton('Save raster coordinates and resolution to csv')
        self.vbox.addWidget(saveCoordsAndRes, 21, 0)

        csv_to_kml = QPushButton('Convert csv points to kml points')
        self.vbox.addWidget(csv_to_kml,22,0)

        
        
        
        #actions
        tile.clicked.connect(lambda: self.tileButton(heightWidth.value(), overlap.value()))
        mosaic.clicked.connect(lambda: self.mosaicButton(resolution.value()))
        slope.clicked.connect(lambda: self.demButton('slope'))
        aspect.clicked.connect(lambda: self.demButton('aspect'))
        hillshade.clicked.connect(lambda: self.demButton('hillshade'))
        roughness.clicked.connect(lambda: self.demButton('roughness'))
        saveCoordsAndRes.clicked.connect(lambda: self.saveCoordsAndResButton())
        csv_to_kml.clicked.connect(lambda: self.csvToKmlButton())
        converter.clicked.connect(lambda: self.converterButton(str(fromType.currentText()),str(toType.currentText())))
        subtract.clicked.connect(lambda: self.subtractButton(first.text(),second.text(), density.text(), mask.text()))
        clipRaster.clicked.connect(lambda: self.clipRasterButton())                                                       
        zonalStats.clicked.connect(lambda: self.zonalStatsButton())
        rasterToShape.clicked.connect(lambda: self.rasterToShapeButton())
        kmeans.clicked.connect(lambda: self.kmeansButton(kmeanClasses.value(), kmeansNoData.text()))
        mergeShapefiles.clicked.connect(lambda: self.mergeShapefilesButton(shape.text()))
        multiband.clicked.connect(lambda: self.multibandButton(numBands.value()))
        virtualRaster.clicked.connect(lambda: self.virtualRasterButton())
        deleter.clicked.connect(lambda: self.deleterButton())

        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(False)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

    def deleterButton(self):
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder of Geotiffs"))
        if folderName1:
            gdal_functions_app.delete_empty_images(folderName1)

    def multibandButton(self, numBands):
        folder_list = []
        for i in range(numBands):
            options1 = QFileDialog.Options()
            options1 |= QFileDialog.DontUseNativeDialog
            folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder of Geotiffs"))
            if folderName1:
                folder_list.append(folderName1)
        if len(folder_list) != 0:
            options1 = QFileDialog.Options()
            options1 |= QFileDialog.DontUseNativeDialog
            folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder to Save to"))
            if folderName1:
                gdal_functions_app.batch_gdal_datacube(folder_list, folderName1)

    def virtualRasterButton(self):
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder of Geotiffs"))
        if folderName1:
            options2 = QFileDialog.Options()
            options2 |= QFileDialog.DontUseNativeDialog
            folderName2 = str(QFileDialog.getExistingDirectory(self, "Select Folder to Save to"))
            if folderName2:
                gdal_functions_app.buildVRT(folderName2, inFolder = folderName1)
                
    def mergeShapefilesButton(self, outShapefile):
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder of Shapefiles"))
        if folderName1:
            gdal_functions_app.mergeShapes(folderName1,outShapefile)

    def kmeansButton(self, classes, noData):
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder of Rasters"))
        if folderName1:
            options2 = QFileDialog.Options()
            options2 |= QFileDialog.DontUseNativeDialog
            folderName2 = str(QFileDialog.getExistingDirectory(self, "Select Folder To Save To"))
            if folderName2:
                gdal_functions_app.kmeans_batch(folderName1, classes, float(noData), folderName2)
                
    def rasterToShapeButton(self):
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder of Rasters"))
        if folderName1:
            gdal_functions_app.raster_to_polygon_batch(folderName1)
            
    def zonalStatsButton(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select shapefile with zones", "","All Files (*);;Shapefile (*.shp)", options=options)
        if fileName:
            options2 = QFileDialog.Options()
            options2 |= QFileDialog.DontUseNativeDialog
            fileName2, _ = QFileDialog.getOpenFileName(self,"Select raster", "","All Files (*);; Geotiff (*.tif)", options=options)
            if fileName2:
                gdal_functions_app.zonalStatsMain(fileName, fileName2)        
    def clipRasterButton(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select raster to clip", "","All Files (*);;Geotiff (*.tif)", options=options)
        if fileName:
            options2 = QFileDialog.Options()
            options2 |= QFileDialog.DontUseNativeDialog
            fileName2, _ = QFileDialog.getOpenFileName(self,"Select shapefile", "","All Files (*);; Shapefile (*.shp)", options=options)
            if fileName2:
                gdal_functions_app.clipRasterToShape(fileName, fileName2)
                
    def converterButton(self,fromType,toType):
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder of Rasters"))
        if folderName1:
            options2 = QFileDialog.Options()
            options2 |= QFileDialog.DontUseNativeDialog
            folderName2 = str(QFileDialog.getExistingDirectory(self, "Select Folder to Save To"))
            if folderName2:
                gdal_functions_app.gdal_convert(folderName1, folderName2, fromType, toType)

    def subtractButton(self,first,second, density, mask):
        if mask != '':
            gdal_functions_app.gdal_difference(first,second,mask=mask)
        else:
            gdal_functions_app.gdal_difference(first,second)
        
    def demButton(self, operation):
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        folderName1 = str(QFileDialog.getExistingDirectory(self, "Select Folder of Rasters"))
        if folderName1:   
            options2 = QFileDialog.Options()
            options2 |= QFileDialog.DontUseNativeDialog
            folderName2 = str(QFileDialog.getExistingDirectory(self, "Select Folder to Save To"))
            if folderName2:
                gdal_functions_app.gdal_dem(folderName1, folderName2, operation)
    def saveCoordsAndResButton(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = str(QFileDialog.getExistingDirectory(self, "Select Folder of Geotiffs"))
        if folderName:
            saveName = os.path.join(folderName,'coords_and_res.csv')
            gdal_functions_app.gdal_get_coords_and_res(folderName, saveName)

    def csvToKmlButton(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select csv", "","All Files (*);;Points File (*.csv)", options=options)
        if fileName:
            gdal_functions_app.csv_to_kml(fileName)
            
        
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

                
## Function outside of the class to run the app   
def run():
    app = QApplication(sys.argv)
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())

## Calling run to run the app
run()
