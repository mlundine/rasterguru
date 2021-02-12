# rasterguru
GUI for various raster tools and a few vector tools.  This GUI will develop more functions as time goes on.

The rationale is to do repetitive GIS tasks quickly.

![GUI PIC](https://github.com/mlundine/rasterguru/blob/master/rastergu_gui.png)

# Setup

Download this repository, unzip it anywhere on your PC.  Or use git to clone it.

To run in Anaconda, the requirements are python 3.7, pyqt5, gdal, pandas, simplekml, scikit-learn, opencv-python, numpy, matplotlib, pykml.

Open up Anaconda prompt or terminal and cd to the rasterguru directory.

cd wherever_you_placed_it/rasterguru

Next make the environment

conda env create --file rasterguru.yml

Now activate that environment

conda activate rasterguru

If you already have a Python 3 environment and just want to install the packages, here are the commands to install the needed packages:

conda install gdal

conda install matplotlib

conda install pandas

conda install numpy

pip install pyqt5

pip install simplekml

pip install opencv-python

pip install pykml

pip install scikit-learn

# Running the GUI in Python

The file to run is Lundine_tile_mosaic_gui.py.

In the terminal, while in the rasterguru directory, while the rasterguru environment is activated, 
you can run the GUI with the command:

python Lundine_tile_mosaic_gui.py

Alternatively, you can import the gdal_functions_app file as a module and run its functions from a Python script.

The code gdal_functions_app.py has many functions that could be useful for
people trying to learn how to work with GDAL and OGR.

# Running the Standalone Executable on a Windows Machine

An executable, RasterGuru.exe is available under Releases.  This is a standalone file that can be used to run the Make Tiles and Make Mosaic functions on Windows machines.

# Making Tiles

When making tiles, the height/width slider can be modified to change the height and width (in pixels) of the output tiles.
If you had an original image with 10 meter resolution, setting height and width to 300 pixels would mean the ouput tiles would have a 3000mx3000m footprint.

Also when making tiles, the overlap slider can be changed to give overlap (in pixels) in the ouput tiles.  If you wanted 300 pixel x 300 pixel tiles, with 50% overlap, overlap would be set to 300/2 = 150.  

# Making Mosaics (Merging Geotiffs)

When making a mosaic, the resolution slider should stay at 1 if you want to retain the same resolution.  If you want to downsample, this value represents the multiple of the original resolution.
For example, if your original image had 10 meter resolution, upping the resolution slider to 2 would output images with 20 meter resolution, upping to 3 would give it 30 meter resolution, and so on.

# Make Multiband Rasters

Make Multiband Rasters will construct new rasters with multiple bands.  You need to specify how many bands.
It will ask for the folders to each band.  Each folder name should differ but the image names inside have matches in every folder.
For example, if you wanted a multiband raster of elevation, slope, and hillshade, you would need to have three folders:
Elevation, Slope, Hillshade.  Inside each folder would be the geotiffs with the specified topographic metrics.
Once you tell it each band's folder, it asks for a folder to save the multiband rasters to.  

# Convert Rasters

Convert Rasters ask for a folder containing geotiffs and a folder to save the results to. You need to specify the from file type and the to file type.

# Slope, Aspect, Roughness, Hillshade

Slope, Aspect, Hillshade, and Roughness ask for a folder containing geotiffs and a folder to save the results to.

# Raster Difference

Raster Difference takes two parameters: The full filepath to the first raster, and the full filepath to the second raster. These must be typed into the text boxes.
The operation would then be first minus second. This button will clip the first raster and resample it to match the extent and resolution of the second raster, and save this to a new geotiff.
It will then calculate the difference and save this as difference.tif. The optional parameter is a filepath to a shapefile to clip the results to. This shapefile should only have one polygon.

# kmeans

kmeans unsupervised classification will perform kmeans unsupervised classification on a folder of geotiffs.  Fill in the number of classes you want, and the nodata value.
Make sure the nodata value is the same fo all of the geotiffs in that folder.  It will ask you for the folder of input images, and also a folder to save the results to.

# Make Virtual Raster Dataset

Make Virtual Raster Dataset makes a text file containing metadata for a folder of rasters.  It is similar to a mosaic,
but it does not create an image.  It is basically a pointer to all of the rasters you combined.

# Clip Raster to Shape

Clip Raster to Shape will clip a raster to the bounds defined by a shapefile.  It asks for a geotiff, and then the shapefile that defines the boundaries.  This shapefile should only have one polygon.

# Conver Rasters to Shapefiles

Convert Rasters to Shapefiles will take as input a folder of rasters with discrete-valued pixels and convert all of these to shapefiles.

# Merge Shapefiles

Merge shapefiles will take as input a folder of shapefiles and put them into one shapefile.  Each shapefile must have identical fields in its attribute table.
Type in the full filepath to the output shapefile before you use this button.  example: C:/data/output/myshapefile.shp.

# Zonal Stats

Zonal Stats asks for a shapefile containing a bunch of polygons and a raster to find mean,median,standard deviation, variance, and range from.
It outputs a csv with these stats for each polygon in the shapefile.

# Save Raster Coords and Resolution

Save raster coords and resolution asks for a folder of geotiffs and then it will save a csv with the image names, raster coordinates, and resolution.

# Convert CSV Points to KML

Convert csv points to kml will take a csv with columns 'name', 'lat', 'long' and then convert this to a kml.


# Extra Scripts

reefmaster_converter.py will convert png/kml pair ReefMaster outputs to geotiffs. 

supervised.py will run various supervised classifiers on an image/mask annotation pair.

# Coming Soon

Projecting Rasters, Some operations on Virtual Raster Datasets, maybe a png/jpeg to geotiff converter where you can enter in the extent/spatial reference.
It all depends on the tasks I come across.






