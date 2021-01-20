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
def gdal_difference(first,second,mask=None):
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
    if mask != None:
        clipRasterToShape(outRaster, mask)
    print('Result: ' + outRaster)
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
        format="GTiff",
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
    
