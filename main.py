import geopandas as gpd
import rasterio as rio
import variete.vrt.vrt
from osgeo import gdal
import tempfile
from pathlib import Path
import warnings
import numba
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.simplefilter("ignore", numba.NumbaDeprecationWarning)
    import xdem

import heerland.main
import heerland.utilities
import heerland.rasters



def main():
    heerland.main.prepare_1990_2010_dems()



if __name__ == "__main__":
    main()
