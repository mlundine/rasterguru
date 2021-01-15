# rasterguru
GUI for tiling and mosaicing rasters

An executable, RasterGuru.exe is available under Releases.  This is a standalone file that can be used to run the application.

There are two functions:
Make Tiles
Make Mosaic

When making tiles, the height/width slider can be modified to change the height and width (in pixels) of the output tiles.
If you had an original image with 10 meter resolution, setting height and width to 300 pixels would mean the ouput tiles would have a 3000mx3000m footprint.

Also when making tiles, the overlap slider can be changed to give overlap (in pixels) in the ouput tiles.  If you wanted 300 pixel x 300 pixel tiles, with 50% overlap, overlap would be set to 300/2 = 150.  

The resolution slider should stay at 1 if you want to retain the same resolution.  If you want to downsample, this value represents the multiple of the original resolution.
For example, if your original image had 10 meter resolution, upping the resolution slider to 2 would output images with 20 meter resolution, upping to 3 would give it 30 meter resolution, and so on.

To run in Anaconda, the requirements are pyqt5, gdal, python 3.7.

conda install gdal

pip install pyqt5

The file to run is Lundine_tile_mosaic_gui.py.




