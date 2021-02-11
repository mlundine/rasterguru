# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:12 2020

@author: Mark Lundine
"""
import sys
import os
import operator
from osgeo import gdal, gdalconst, ogr, osr
import osgeo.gdalnumeric as gdn
import numpy as np
import glob
import pandas as pd
import simplekml
import subprocess
import shutil
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from pykml import parser
#import supervised
# =============================================================================
# get coords and res will make a spreadsheet of the coordinates and resolution for a folder
# need to specify the folder with the DEMs and a .csv file path to save the DEMs' coordinates and resolutions
# of DEMs, using arcpy.  
# =============================================================================
def gdal_get_coords_and_res(folder, saveFile):
    """
    Takes a folder of geotiffs and outputs a csv with bounding box coordinates and x and y resolution
    inputs:
    folder (string): filepath to folder of geotiffs
    saveFile (string): filepath to csv to save to
    """
    
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
    """
    Converts geotiffs and erdas imagine images to .tif,.jpg, .png, or .img
    inputs:
    inFolder (string): folder of .tif or .img images
    outFolder (string): folder to save result to
    inType (string): extension of input images ('.tif' or '.img')
    outType (string): extension of output images ('.tif', '.img', '.jpg', '.png')
    """
    
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
    """
    Usess gdal demprocessing to compute slope, hillshade, roughness, or aspect
    inputs:
    inFolder (string): folder of geotiffs
    outFolder (string): folder to save result geotiffs so
    computation (string): 'hillshade', 'slope', 'Roughness', or 'aspect'
    """
    for dem in glob.glob(inFolder + '/*.tif'):
        name = os.path.basename(dem)
        print(name)
        outDEM = os.path.join(outFolder, name)
        gdal.DEMProcessing(outDEM, dem, computation)
        

##Converts csvs to kmls
##Columns of csvs must be 'name','lat','long'
def csv_to_kml(inFile):
    """
    Converts a csv to a kml
    kml will save in same directory as csv
    csv columns must be 'name', 'lat', and 'long'
    inputs:
    inFile (string): filepath to csv to convert to kml
    """
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


def gdal_datacube(images, outFolder):
    """
    takes a list of single band geotiffs with matching extent and resolution
    constructs multiband geotiff, saves result with 'cube' appendd to the end
    inputs:
    images: the list of filepaths to the images
    outFolder: the filepath of the folder to save to
    """
    ##assign name for output file
    name = os.path.splitext(os.path.basename(images[0]))[0] + 'cube.tif'
    out_path = os.path.join(outFolder, name)
    print(out_path)
    
    ##see how many bands, rows, and columns output will have, get datatype
    nbands = len(images)
    firstBand = gdal.Open(images[0])
    rows,cols = firstBand.RasterYSize, firstBand.RasterXSize
    datatype = firstBand.GetRasterBand(1).DataType

    ##establish geotransform and projection
    geotrans = firstBand.GetGeoTransform()
    proj = firstBand.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    
    ##close first image
    firstBand = None
    
    ##initialize datacube array, then fill each band with geotiff data
    dataCube = np.empty((rows, cols, nbands))
    for i in range(len(images)):
        band = gdal.Open(images[i])
        band = band.GetRasterBand(1).ReadAsArray()
        dataCube[:,:,i] = band
        band = None
    ## save geotiff
    driverTiff = gdal.GetDriverByName('GTiff')
    out_tiff = driverTiff.Create(out_path, cols, rows, nbands, datatype)
    out_tiff.SetGeoTransform(geotrans)
    out_tiff.SetProjection(srs.ExportToWkt())
    for i in range(1,nbands+1):
        out_tiff.GetRasterBand(i).SetNoDataValue(-9999)
        out_tiff.GetRasterBand(i).WriteArray(dataCube[:,:,i-1])

    ## clean up
    out_tiff = None
    datacube = None

def batch_gdal_datacube(inFolders, outFolder):
    """
    runs gdal_datacube on folders of geotiffs
    inputs:
    inFolders: list of folders containing geotiffs,
    foldernames must be different but image names much match
    outFolder: filepath to save the datacubes
    """
    path, dirs, files = next(os.walk(inFolders[0]))
    file_count = len(files)
    stacked_images = [None]*len(inFolders)
    for i in range(len(inFolders)):
        im_list = [None]*file_count
        j=0
        for image in glob.glob(inFolders[i]+'/*.tif'):
            im_list[j]=image
            j=j+1
        im_list = sorted(im_list)
        stacked_images[i] = im_list
    stacked_images = np.array(stacked_images)
    for i in range(file_count):
        image_list = stacked_images[:,i]
        gdal_datacube(image_list, outFolder)

def getMatchingExtentAndRes(rasterToResize, resizerRaster):
    """
    Resizes and resamples a raster to the extent and resolution
    of another raster. The resized raster gets saved to the same
    directory as rasterToResize with "newExtent" added on to the name.
    inputs:
    rasterToResize: filepath to the geotiff you want to resize
    resizerRaster: filepath to the geotiff containing
    the wanted extent and resolution
    """
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
def gdal_difference(first,second,mask=None):
    """
    Computes the difference between two rasters
    first minus second
    Saves result as difference.tif to directory of first raster.
    inputs:
    first (string): filepath to first raster
    second (string): filepath to second raster
    mask (string): optional parameter, a path to
    a shapefile to clip the difference to.
    """
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
##        second_dem = None
##        second_arr = None
        newFirst = getMatchingExtentAndRes(first,second)
##        newSecond = getMatchingExtentAndRes(second,first)
        print('New First Raster Saved To: '+newFirst)
##        print('New Second Raster Saved To: '+newSecond)
        first_dem = gdal.Open(newFirst)
        first_arr = first_dem.GetRasterBand(1).ReadAsArray().astype('float32')
##        second_dem = gdal.Open(newSecond)
##        second_arr = second_dem.GetRasterBand(1).ReadAsArray().astype('float32')
              

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
    if mask != None:
        clipRasterToShape(outRaster, mask)
    print('Result: ' + outRaster)
    if mask != None:
        print('Clipped Result: ' + os.path.splitext(outRaster)[0]+'clipped.tif')

def reproject_vector(in_path, out_path, dest_srs):
    """
    Reprojects a vector file to a new SRS. Simple wrapper for ogr2ogr.
    Parameters
    ----------
    in_path
    out_path
    dest_srs

    """
    subprocess.run(["ogr2ogr", out_path, in_path, '-t_srs', dest_srs.ExportToWkt()])
    
def clipRasterToShape(raster_path, aoi_path, srs_id=4326, flip_x_y = False):
    """
    Clips a raster at raster_path to a shapefile given by aoi_path. Assumes a shapefile only has one polygon.
    Will np.floor() when converting from geo to pixel units and np.absolute() y resolution form geotransform.
    Will also reproject the shapefile to the same projection as the raster if needed.

    Parameters
    ----------
    raster_path
        Path to the raster to be clipped.
    aoi_path
        Path to a shapefile containing a single polygon
    out_path
        Path to a location to save the final output raster

    """
    # https://gis.stackexchange.com/questions/257257/how-to-use-gdal-warp-cutline-option
    td = os.path.dirname(raster_path)
    raster = gdal.Open(raster_path)
    in_gt = raster.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjection())
    intersection_path = os.path.join(td, 'intersection')
    aoi = ogr.Open(aoi_path)
    if aoi.GetLayer(0).GetSpatialRef().ExportToWkt() != srs.ExportToWkt():    # Gross string comparison. Might replace with wkb
        aoi = None
        tmp_aoi_path = os.path.join(td, "tmp_aoi.shp")
        reproject_vector(aoi_path, tmp_aoi_path, srs)
        aoi = ogr.Open(tmp_aoi_path)
    intersection = get_aoi_intersection(raster, aoi)
    min_x_geo, max_x_geo, min_y_geo, max_y_geo = intersection.GetEnvelope()
    if flip_x_y:
        min_x_geo, min_y_geo = min_y_geo, min_x_geo
        max_x_geo, max_y_geo = max_y_geo, max_x_geo
    width_pix = int(np.floor(max_x_geo - min_x_geo)/in_gt[1])
    height_pix = int(np.floor(max_y_geo - min_y_geo)/np.absolute(in_gt[5]))
    new_geotransform = (min_x_geo, in_gt[1], 0, max_y_geo, 0, in_gt[5])   # OK, time for hacking
    write_geometry(intersection, intersection_path, srs_id=srs.ExportToWkt())
    clip_spec = gdal.WarpOptions(
        format="VRT",
        cutlineDSName=intersection_path+r"/geometry.shp",   # TODO: Fix the need for this
        cropToCutline=True,
        width=width_pix,
        height=height_pix,
        dstSRS=srs
    )
    out_path = os.path.splitext(raster_path)[0]+'clipped.tif'
    out = gdal.Warp(out_path, raster, options=clip_spec)
    out.SetGeoTransform(new_geotransform)
    out = None
    shutil.rmtree(intersection_path)
    return(out_path)

def get_aoi_intersection(raster, aoi):
    """
    Returns a wkbPolygon geometry with the intersection of a raster and a shpefile containing an area of interest

    Parameters
    ----------
    raster
        A raster containing image data
    aoi
        A shapefile with a single layer and feature
    Returns
    -------
    a ogr.Geometry object containing a single polygon with the area of intersection

    """
    raster_shape = get_raster_bounds(raster)
    aoi.GetLayer(0).ResetReading()  # Just in case the aoi has been accessed by something else
    aoi_feature = aoi.GetLayer(0).GetFeature(0)
    aoi_geometry = aoi_feature.GetGeometryRef()
    return aoi_geometry.Intersection(raster_shape)

def get_raster_bounds(raster):
    """
    Returns a wkbPolygon geometry with the bounding rectangle of a raster calculated from its geotransform.

    Parameters
    ----------
    raster
        A gdal.Image object

    Returns
    -------
    An ogr.Geometry object containing a single wkbPolygon with four points defining the bounding rectangle of the
    raster.

    Notes
    -----
    Bounding rectangle is obtained from raster.GetGeoTransform(), with the top left corners rounded
    down to the nearest multiple of of the resolution of the geotransform. This is to avoid rounding errors in
    reprojected geotransformations.
    """
    raster_bounds = ogr.Geometry(ogr.wkbLinearRing)
    geotrans = raster.GetGeoTransform()
    # We can't rely on the top-left coord being whole numbers any more, since images may have been reprojected
    # So we floor to the resolution of the geotransform maybe?
    top_left_x = floor_to_resolution(geotrans[0], geotrans[1])
    top_left_y = floor_to_resolution(geotrans[3], geotrans[5]*-1)
    width = geotrans[1]*raster.RasterXSize
    height = geotrans[5]*raster.RasterYSize * -1  # RasterYSize is +ve, but geotransform is -ve
    raster_bounds.AddPoint(top_left_x, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(raster_bounds)
    return bounds_poly
def floor_to_resolution(input, resolution):
    """
    Returns input rounded DOWN to the nearest multiple of resolution. Used to prevent float errors on pixel boarders.

    Parameters
    ----------
    input
        The value to be rounded
    resolution
        The resolution

    Returns
    -------
    The largest value between input and 0 that is divisible by resolution.

    Notes
    -----
    Uses the following formula: input-(input%resolution)


    """
    if resolution > 1:
        return input - (input%resolution)
    else:
        log.warning("Low resolution detected, assuming in degrees. Rounding to 6 dp.\
                Probably safer to reproject to meters projection.")
        resolution = resolution * 1000000
        input = input * 1000000
        return (input-(input%resolution))/1000000
def write_geometry(geometry, out_path, srs_id=4326):
    """
    Saves the geometry in an ogr.Geometry object to a shapefile.

    Parameters
    ----------
    geometry
        An ogr.Geometry object
    out_path
        The location to save the output shapefile
    srs_id
        The projection of the output shapefile. Can be an EPSG number or a WKT string.

    Notes
    -----
    The shapefile consists of one layer named 'geometry'.


    """
    # TODO: Fix this needing an extra filepath on the end
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(out_path)
    srs = osr.SpatialReference()
    if type(srs_id) is int:
        srs.ImportFromEPSG(srs_id)
    if type(srs_id) is str:
        srs.ImportFromWkt(srs_id)
    layer = data_source.CreateLayer(
        "geometry",
        srs,
        geom_type=geometry.GetGeometryType())
    feature_def = layer.GetLayerDefn()
    feature = ogr.Feature(feature_def)
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)
    data_source.FlushCache()
    data_source = None

def zonal_stats(feat, input_zone_polygon, input_value_raster):

    # Open data
    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    # Read raster as arrays
    banddataraster = raster.GetRasterBand(1)
    dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)

    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

    # Mask zone of raster
    zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))

    # Calculate statistics of zonal raster
    return np.mean(zoneraster),np.median(zoneraster),np.std(zoneraster),np.var(zoneraster),(np.max(zoneraster)-np.min(zoneraster))


def loop_zonal_stats(input_zone_polygon, input_value_raster):

    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())
    means = [None]*len(featList)
    medians = [None]*len(featList)
    stds = [None]*len(featList)
    variances = [None]*len(featList)
    ranges = [None]*len(featList)
    i=0
    for FID in featList:
        feat = lyr.GetFeature(FID)
        meanValue = zonal_stats(feat, input_zone_polygon, input_value_raster)
        means[i] = meanValue[0]
        medians[i] = meanValue[1]
        stds[i] = meanValue[2]
        variances[i] = meanValue[3]
        ranges[i] = meanValue[4]
        i=i+1
    statDict = {'FID':featList,
                'Mean':means,
                'Median':medians,
                'Std':stds,
                'Var':variances,
                'Range':ranges
                }
    statDictdf = pd.DataFrame.from_dict(statDict)
    return statDictdf

def zonalStatsMain(input_zone_polygon, input_value_raster):
    """
    Computes zonal statistics (mean, median, std, variance, range)
    Saves results to csv
    inputs:
    input_zone_polygon: shapefile with polygon zones
    input_value_raster: raster to perform statistics on
    """
    df = loop_zonal_stats(input_zone_polygon, input_value_raster)
    savefile = os.path.join(os.path.splitext(input_zone_polygon)[0]+'_zonalstats.csv')
    df.to_csv(savefile,sep=',',index=False)

def plotDEM(dem_list, clim=None, titles=None, cmap='gray', label=None, overlay=None, fn=None):
    """
    Plots DEMs
    inputs:
    dem_list: a list of geotiff DEMs
    clim: optional, elevation limits (min,max)
    titles: optional, a list of titles for the DEMs
    cmap: optional, matplotlib colormap, default is grayscale
    label: optional, label for colorbar ex: 'Elevation (m)'
    overlay: optional, list of hillshade rasters that correspond to list of DEMs
    fn: optional, png path to save plot to
    """
    arr_list = [None]*len(dem_list)
    i=0
    for dem in dem_list:
        dem = gdal.Open(dem)
        dem_arr = np.array(dem.GetRasterBand(1).ReadAsArray().astype('float32'))
        arr_list[i]=dem_arr
        i=i+1
    if overlay != None:
        over_arr_list = [None]*len(overlay)
        i=0
        for dem in overlay:
            dem = gdal.Open(dem)
            dem_arr = np.array(dem.GetRasterBand(1).ReadAsArray().astype('uint16'))
            over_arr_list[i]=dem_arr
            i=i+1
    else:
        pass
    fig, axa = plt.subplots(1,len(dem_list), sharex=True, sharey=True, figsize=(10,5))
    alpha = 1.0
    try:
        for n, ax in enumerate(axa):
            #Gray background
            ax.set_facecolor('0.5')
            #Force aspect ratio to match images
            ax.set(adjustable='box', aspect='equal')
            #Turn off axes labels/ticks
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if titles is not None:
                ax.set_title(titles[n])
            #Plot background shaded relief map
            if overlay is not None:
                alpha = 0.7
                axa[n].imshow(over_arr_list[n], cmap='gray', clim=(1,255))
        #Plot each array
        im_list = [axa[i].imshow(arr_list[i], clim=clim, cmap=cmap, alpha=alpha) for i in range(len(arr_list))]
        fig.tight_layout()
        fig.colorbar(im_list[0], ax=axa.ravel().tolist(), label=label, extend='both', shrink=0.5)
        if fn is not None:
            fig.savefig(fn, bbox_inches='tight', pad_inches=0, dpi=150)   
    except:
        #Gray background
        axa.set_facecolor('0.5')
        #Force aspect ratio to match images
        axa.set(adjustable='box', aspect='equal')
        #Turn off axes labels/ticks
        axa.get_xaxis().set_visible(False)
        axa.get_yaxis().set_visible(False)
        if titles is not None:
            axa.set_title(titles[0])
        #Plot background shaded relief map
        if overlay is not None:
            alpha = 0.7
            axa.imshow(over_arr_list[0], cmap='gray', clim=(1,255))
        #Plot each array
        im_list = [axa.imshow(arr_list[i], clim=clim, cmap=cmap, alpha=alpha) for i in range(len(arr_list))]
        fig.tight_layout()
        fig.colorbar(im_list[0], ax=axa, label=label, extend='both', shrink=0.5)
        if fn is not None:
            fig.savefig(fn, bbox_inches='tight', pad_inches=0, dpi=150)
            
def raster_to_polygon(raster_path):
    """
    Converts raster with discrete pixel values to polygons
    inputs:
    raster_path: the input raster filepath
    """
    
    shape_path = os.path.splitext(raster_path)[0]+'poly.shp'
    os.system('python gdal_polygonize.py ' + raster_path + ' ' + shape_path)

def raster_to_polygon_batch(folder):
    """
    Converts a folder of rasters to shapefiles
    inputs:
    folder: filepath to folder of geotiffs
    """
    for raster in glob.glob(folder + '/*.tif'):
        print(raster)
        raster_to_polygon(raster)

def kmeans(image_path, classes, geo, no_data_val, show, save_folder = None, detect=False):
    """
    performs kmeans unsupervised clustering on an image,
    saves result to a jpeg or geotiff
    inputs:
    image_path: path to an image (.jpeg, .tif, .img, .png)
    classes: number of classes to cluster image into
    geo: True if the images are georeferenced (for geotiffs and erdas imagines)
    False if they are not georeferenced
    no_data_val: only for georeferenced images, the no-data value
    show: only for non-georeferenced images, just will show the result
    """
    save_path = os.path.splitext(image_path)[0]+'kmeans'+str(classes)
    if save_folder != None:
        save_path = os.path.join(save_folder, os.path.basename(save_path))
        
    if geo == True:
        
        ### read in image to classify with gdal
        driverTiff = gdal.GetDriverByName('GTiff')
        input_raster = gdal.Open(image_path)
        nbands = input_raster.RasterCount
        nodata = no_data_val
        ### create an empty array, each column of the empty array will hold one band of data from the image
        ### loop through each band in the image nad add to the data array
        data = np.empty((input_raster.RasterXSize*input_raster.RasterYSize, nbands))
        for i in range(1, nbands+1):
            band = input_raster.GetRasterBand(i).ReadAsArray()
            data[:, i-1] = band.flatten()
        data[data<nodata] = np.mean(data)
    else:
        save_path = save_path + '.jpeg'
        im = cv2.imread(image_path)
        rows,cols,nbands = np.shape(im)
        data = np.empty((rows*cols,nbands))
        for i in range(0,nbands):
            band = im[:,:,i]
            data[:, i-1] = band.flatten()
            
    
    # set up the kmeans classification, fit, and predict
    km = KMeans(n_clusters=classes)
    km.fit(data)
    km.predict(data)

    if geo==True:
        save_path = save_path+'.tif'
        # format the predicted classes to the shape of the original image
        out_dat = km.labels_.reshape((input_raster.RasterYSize, input_raster.RasterXSize))
        if detect==True:
            val = out_dat[int(input_raster.RasterYSize/2),int(input_raster.RasterXSize/2)]
            a, counts = np.unique(out_dat, return_counts=True)
            idx = np.argmax(counts)
            mode = a[idx]
            if val == 1 and mode == 1:
                print('hi')
                pass
            elif val == 0 and mode == 0:
                print('hello')
                new_out_dat = np.empty(np.shape(out_dat),dtype=int)
                idx1 = out_dat[out_dat<1]
                idx2 = out_dat[out_dat>0]
                new_out_dat[idx1] = 1
                new_out_dat[idx2] = 0
                out_dat = new_out_dat
                new_out_dat = None
                if np.sum(np.ravel(out_dat)) < 5:
                    return
            else:
                'skipped'
                return
        # save the original image with gdal
        clfds = driverTiff.Create(save_path, input_raster.RasterXSize, input_raster.RasterYSize, 1, gdal.GDT_Float32)
        clfds.SetGeoTransform(input_raster.GetGeoTransform())
        clfds.SetProjection(input_raster.GetProjection())
        clfds.GetRasterBand(1).SetNoDataValue(0)
        clfds.GetRasterBand(1).WriteArray(out_dat)
        clfds = None
        out_dat=None
    else:
        # format the predicted classes to the shape of the original image
        out_dat = km.labels_.reshape((rows,cols))
        cv2.imwrite(save_path, out_dat)
        if show==True:
            plt.imshow(out_dat)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.show()
        else:
            pass

def kmeans_batch(inFolder, numClasses, noDataVal, outFolder, detect=False):
    """
    Performs kmeans clustering on a folder of geotiffs or erdas imagine images
    inputs:
    inFolder: path to input folder
    numClasses: integer number of classes
    noDataVal: the value for nodata in the rasters
    outFolder: path to the output folder
    """
    for image in glob.glob(inFolder + '/*.tif'):
        kmeans(image, numClasses, True, noDataVal, False, save_folder=outFolder, detect = detect)
    for image in glob.glob(inFolder + '/*.img'):
        kmeans(image, numClasses, True, noDataVal, False, save_folder=outFolder)

def clipRasterToBbox(rasterToClip, bbox, saveFile):
    """
    Clips a raster to new bounding box
    inputs:
    rasterToClip: filepath to raster to clip
    bbox: [xmin, ymax, xmax, ymin]
    savFile: raster filepath to save to
    """
    ds = gdal.Open(rasterToClip)
    ds = gdal.Translate(saveFile, ds, projWin = bbox)
    ds = None
    
def batchClipRasterToBbox(bboxCSV, inFolder, outFolder):
    """
    clips a folder of rasters to bounding boxes defined in a csv
    inputs:
    bboxCSV: the csv containing the bounding boxes and corresponding filenames
    stucture must be filename, xmin, ymin, xmax, ymax
    inFolder: the folder containing the rasters
    outFolder: folder to save clipped rasters to
    the saved rasters will be given a numeric id
    """
    df = pd.read_csv(bboxCSV)
    ID = 1
    for i in range(len(df)):
        filename = df.iloc[i,0]
        xmin = df.iloc[i,1]
        ymin = df.iloc[i,2]
        xmax = df.iloc[i,3]
        ymax = df.iloc[i,4]
        width = xmax-xmin
        height = ymax-ymin
        bbox = [xmin,ymin,xmax,ymax]
        image = os.path.join(inFolder, filename+'.tif')
        saveImage = os.path.join(outFolder, filename+str(ID)+'.tif')
        print(saveImage)
        clipRasterToBbox(image, bbox, saveImage)
        ID = ID+1

def filterGeobox(bboxCSV, threshold):
    """
    Filters geobbox csv to a specific threshold
    inputs:
    bboxcsv: filepath to the geobbox csv
    threshold: threshold value to filter above
    """
    saveCSV = os.path.splitext(bboxCSV)[0]+'filter.csv'
    df = pd.read_csv(bboxCSV)
    querystr = "score >= "+str(threshold)
    filtered = df.query(querystr)
    filtered = filtered.reset_index(drop=True)
    filtered.to_csv(saveCSV,index=False)

def mergeShapes(folder, outShape):
    """
    Merges a bunch of shapefiles. Sshapefiles have to have same fields
    in attribute table.
    inputs:
    folder: filepath to folder with all of the shapefiles
    outShape: filepath to file to save to, has to have .shp extension.
    """
    os.system('python MergeSHPfiles_cmd.py ' + folder + ' ' + outShape)


def buildVRT(outFolder, inFolder=None, rasterList = None):
    vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
    if inFolder != None:
        image_list = []
        for image in glob.glob(inFolder + '/*.tif'):
            image_list.append(image)
    if rasterList != None:
        image_list = rasterList
        
    saveFile = os.path.join(outFolder, 'mosaic.vrt')
    gdal.BuildVRT(saveFile, image_list, options=vrt_options)

def clipVRT(vrt_path, clipShape):
    inraster = vrt_path
    inshape = clipShape
    ds = ogr.Open(inshape)
    lyr = ds.GetLayer()
    

    featList = range(lyr.GetFeatureCount())
    for FID in featList:
        feat = lyr.GetFeature(FID)
        ID = feat.GetFieldAsString('Name')
        print(str(FID/featList[-1]))
        outraster = os.path.splitext(vrt_path)[0]+ID+'.tif'
        if os.path.isfile(outraster):
            continue
        query = "'Name'=%s" % ID
        subprocess.call(['gdalwarp', inraster, outraster, '-cutline', inshape, 
                         '-crop_to_cutline', '-cwhere', "Name = %s" % ID])

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

