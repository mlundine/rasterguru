###Import libraries
from pykml import parser
from osgeo import gdal, osr
import numpy as np
import cv2
import glob
import os

def pngToGeotiff(png_path, kml_path, output_path):
    """
    Converts Reefmaster output (png and kml) to geotiff
    inputs:
    png_path: path to the png containing image data
    kml_path: path to the kml containing spatial extent
    output_path: path to save geotiff to (ends with .tif)
    """

    ##open png as numpy array, get shape
    image = cv2.imread(png_path)
    rows,cols,bands = np.shape(image)

    ## get the geotransform and coordinate system from the kml
    geotransform, proj = parseKML(kml_path, rows, cols)

    ## save geotiff
    driverTiff = gdal.GetDriverByName('GTiff')
    out_tiff = driverTiff.Create(output_path, cols, rows, bands, gdal.GDT_Int16)
    out_tiff.SetGeoTransform(geotransform)
    out_tiff.SetProjection(proj.ExportToWkt())
    for i in range(1,bands+1):
        out_tiff.GetRasterBand(i).SetNoDataValue(-9999)
        out_tiff.GetRasterBand(i).WriteArray(image[:,:,i-1])

    ## clean up
    out_tiff = None
    image = None
    
def parseKML(kml_path, nrows, ncols, epsg_code=4326):
    """
    reads the xmin,xmax,ymin,and ymax from reefmaster kml
    inputs:
    kml_path: filepath to the kml
    epsg_code (optional): in case the data is not in wgs84 lat lon
    outputs:
    geotransform: gdal geotransform object
    proj: gdal projection object
    """
    with open(kml_path) as f:
        tree = parser.parse(f)
        root = tree.getroot()
    

    ymax = float(root.Document.GroundOverlay.LatLonBox.north.text)
    ymin = float(root.Document.GroundOverlay.LatLonBox.south.text)
    xmax = float(root.Document.GroundOverlay.LatLonBox.east.text)
    xmin = float(root.Document.GroundOverlay.LatLonBox.west.text)

    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0,-yres)
    proj = osr.SpatialReference()                 # Establish its coordinate encoding
    proj.ImportFromEPSG(epsg_code)                     # This one specifies WGS84 lat long.
    
    return geotransform, proj

def batchPNGtoGeotiff(folder):
    """
    runs pngToGeotiff on a folder of png/kml pairs
    saves geotiffs to same folder
    inputs:
    folder: filepath to the folder the png/kml pairs are sitting in
    """
    for image in glob.glob(folder + '/*.png'):
        base = os.path.splitext(image)[0]
        kml = base + '.kml'
        out = base + '.tif'
        pngToGeotiff(image, kml, out)
    
