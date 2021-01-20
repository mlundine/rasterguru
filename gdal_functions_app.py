# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:12 2020

@author: Mark Lundine
"""
import sys
import os
from osgeo import gdal, gdalconst
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

def getMatchingExtentAndRes(rasterToResize, resizerRaster):
    # Source
    src_filename = rasterToResize
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # We want a section of source that matches this:
    match_filename = resizerRaster
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst_filename = os.path.splitext(rasterToResize)[0]+'newExtent.tif'
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
    src = None
    math_ds = None
    dst = None
    return(dst_filename)

def write_gtiff(array, gdal_obj, outputpath, dtype=gdal.GDT_UInt16, options=0, color_table=0, nbands=1, nodata=False):
    """
    Writes a geotiff.

    array: numpy array to write as geotiff
    gdal_obj: object created by gdal.Open() using a tiff that has the SAME CRS, geotransform, and size as the array you're writing
    outputpath: path including filename.tiff
    dtype (OPTIONAL): datatype to save as
    nodata (default: FALSE): set to any value you want to use for nodata; if FALSE, nodata is not set
    """

    gt = gdal_obj.GetGeoTransform()

    width = np.shape(array)[1]
    height = np.shape(array)[0]

    # Prepare destination file
    driver = gdal.GetDriverByName("GTiff")
    if options != 0:
        dest = driver.Create(outputpath, width, height, nbands, dtype, options)
    else:
        dest = driver.Create(outputpath, width, height, nbands, dtype)

    # Write output raster
    if color_table != 0:
        dest.GetRasterBand(1).SetColorTable(color_table)

    dest.GetRasterBand(1).WriteArray(array)

    if nodata is not False:
        dest.GetRasterBand(1).SetNoDataValue(nodata)

    # Set transform and projection
    dest.SetGeoTransform(gt)
    wkt = gdal_obj.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())

    # Close output raster dataset 
    dest = None
    
##TODO add a mask polygon option
## need to have 
def gdal_difference(first,second,density=None, mask=None):
    #open dems
    first_dem = gdal.Open(first)
    second_dem = gdal.Open(second)

    #convert to arrays, get shape
    first_arr = first_dem.GetRasterBand(1).ReadAsArray().astype('float32')
    first_size = first_arr.shape
    second_arr = second_dem.GetRasterBand(1).ReadAsArray().astype('float32')
    second_size = second_arr.shape

    #get projections and geotransforms
    first_prj = first_dem.GetProjection()
    second_prj = second_dem.GetProjection()
    first_geotransform = first_dem.GetGeoTransform()
    second_geotransform = second_dem.GetGeoTransform()

    #check if size, projections, and geotranforms match, else make them match
    if (first_prj == second_prj) and (first_size == second_size) and (first_geotransform == second_geotransform):
        pass
    else:
        first_dem = None
        first_arr = None
        newFirst = getMatchingExtentAndRes(first,second)
        print('New Second Raster Saved To: '+newFirst)
        first_dem = gdal.Open(newFirst)
        first_arr = first_dem.GetRasterBand(1).ReadAsArray().astype('float32')

    #calculate difference
    result = first_arr - second_arr

    #get shape, datatype, and coordinate system
    rows,cols = result.shape
    geoTransform = first_dem.GetGeoTransform()
    band=first_dem.GetRasterBand(1)
    datatype=band.DataType

    #write the raster
    driver=first_dem.GetDriver()
    folder = os.path.dirname(first)
    outRaster = os.path.join(folder, 'difference.tif')
    outDS = driver.Create(outRaster, cols,rows, 1,datatype)
    geoTransform = first_dem.GetGeoTransform()
    outDS.SetGeoTransform(geoTransform)
    outDS.SetProjection(second_prj)
    outBand = outDS.GetRasterBand(1)
    outBand.WriteArray(result, 0, 0)

    #clean up
    outDS=None
    first_dem=None
    second_dem=None
    first_arr=None
    second_arr=None
    result=None
    band=None
    print('Result: ' + outRaster)

