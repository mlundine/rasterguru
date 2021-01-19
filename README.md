# rasterguru
GUI for various raster tools and a few vector tools.  This GUI will develop more functions as time goes on.

The rationale is to do repetitive GIS tasks quickly.

![GUI PIC](https://github.com/mlundine/rasterguru/blob/master/rastergu_gui.png)

When making tiles, the height/width slider can be modified to change the height and width (in pixels) of the output tiles.
If you had an original image with 10 meter resolution, setting height and width to 300 pixels would mean the ouput tiles would have a 3000mx3000m footprint.

Also when making tiles, the overlap slider can be changed to give overlap (in pixels) in the ouput tiles.  If you wanted 300 pixel x 300 pixel tiles, with 50% overlap, overlap would be set to 300/2 = 150.  

When making a mosaic, the resolution slider should stay at 1 if you want to retain the same resolution.  If you want to downsample, this value represents the multiple of the original resolution.
For example, if your original image had 10 meter resolution, upping the resolution slider to 2 would output images with 20 meter resolution, upping to 3 would give it 30 meter resolution, and so on.

Convert Rasters, Slope, Aspect, Hillshade, and Roughness ask for a folder containing geotiffs and a folder to save the results to.

Save raster coords and resolution asks for a folder of geotiffs and then it will save a csv with the image names, raster coordinates, and resolution.

Convert csv points to kml will take a csv with columns 'name', 'lat', 'long' and then convert this to a kml.

Coming soon: Raster Differencing, Constructing Multiband Rasters.

To run in Anaconda, the requirements are python 3.7, pyqt5, gdal, pandas, simplekml.

conda install gdal

pip install pyqt5

conda install pandas

pip install simplekml

The file to run is Lundine_tile_mosaic_gui.py.

An executable, RasterGuru.exe is available under Releases.  This is a standalone file that can be used to run the Make Tiles and Make Mosaic functions on Windows machines.


