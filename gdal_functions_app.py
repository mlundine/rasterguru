# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:12 2020

@author: Mark Lundine
"""
import sys
import os
from osgeo import gdal
import osgeo.gdalnumeric as gdn
import numpy as np
import glob
import pandas as pd
import simplekml
# =============================================================================
# get coords and res will make a spreadsheet of the coordinates and resolution for a folder
# need to specify the folder with the DEMs and a .csv file path to save the DEMs' coordinates and resolutions
# of DEMs, using arcpy.  
# =============================================================================
def gdal_get_coords_and_res(folder, saveFile):
    myList = []
    myList.append(['file', 'xmin', 'ymin', 'xmax', 'ymax', 'xres', 'yres'])
    for dem in glob.glob(folder + '/*.tif'):
        src = gdal.Open(dem)
        xmin, xres, xskew, ymax, yskew, yres  = src.GetGeoTransform()
        xmax = xmin + (src.RasterXSize * xres)
        ymin = ymax + (src.RasterYSize * yres)
        myList.append([dem, xmin, ymin, xmax, ymax, xres, -yres])
        src = None
    np.savetxt(saveFile, myList, delimiter=",", fmt='%s')

#converts between .tif, .img, .jpg, .npy, .png
#TODO make output files have spatial reference
def gdal_convert(inFolder, outFolder, inType, outType):
    for im in glob.glob(inFolder + '/*'+inType):
        print('in: ' + im)
        imName = os.path.splitext(os.path.basename(im))[0]
        outIm = os.path.join(outFolder, imName+outType)
        print('out: ' + outIm)
        if outType == '.npy':
            raster = gdal.Open(im)
            bands = [raster.GetRasterBand(i) for i in range(1, raster.RasterCount+1)]
            arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype('float32')
            arr = np.transpose(arr, [1,2,0])
            np.save(outIm, arr)
        else:
            raster = gdal.Open(im)
            gdal.Translate(outIm, raster)
            raster = None

#calculates slope, aspect, hillshade, or roughness on a folder of geotiffs
#TODO make output files have spatial reference
def gdal_dem(inFolder, outFolder, computation):
    for dem in glob.glob(inFolder + '/*.tif'):
        name = os.path.basename(dem)
        print(name)
        outDEM = os.path.join(outFolder, name)
        gdal.DEMProcessing(outDEM, dem, computation)    

##Converts csvs to kmls
##Columns of csvs must be 'name','lat','long'
def csv_to_kml(inFile):
    outFile = os.path.splitext(inFile)[0]+'.kml'
    kml = simplekml.Kml()
    df = pd.read_csv(inFile)
    for i in range(len(df)):
        ptName = df.at[i,'name']
        ptLat = df.at[i, 'lat']
        ptLong = df.at[i, 'long']
        pnt = kml.newpoint(name=ptName, coords=[(ptLong, ptLat)])
        pnt.style.iconstyle.scale=1
    kml.save(outFile)

##TODO
#def gdal_datacube(inFolders,outFolder):
    
##TODO
## need to have 
def gdal_difference(first,second,density=None, mask=None):
    first_dem = gdal.Open(first)
    second_dem = gdal.Open(second)
    result = np.array(first_dem.GetRasterBand(1).ReadAsArray())-np.array(second_dem.GetRasterBand(1).ReadAsArray())
    rows,cols = result.shape
    
    geoTransform = first_dem.GetGeoTransform()
    band=first_dem.GetRasterBand(1)
    datatype=band.DataType
    proj = first_dem.GetProjection()
    
    ############write raster##########
    
    driver=first_dem.GetDriver()
    folder = os.path.dirname(first)
    outRaster = os.path.join(folder, 'difference.tif')
    
    outDS = driver.Create(outRaster, cols,rows, 1,datatype)
    geoTransform = first_dem.GetGeoTransform()
    outDS.SetGeoTransform(geoTransform)
    proj = first_dem.GetProjection()
    outDS.SetProjection(proj)
    outBand = outDS.GetRasterBand(1)
    outBand.WriteArray(result, 0, 0)
    #data is the output array to written in tiff file
    outDS=None
    
    
        
