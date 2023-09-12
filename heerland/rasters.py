
from osgeo import gdal
import rasterio as rio
from pathlib import Path
import xdem
import geopandas as gpd
import rasterio as rio
import variete.vrt.vrt
from osgeo import gdal
import tempfile
from pathlib import Path
import warnings
import numba

def load_raster_subset(
    filepath: Path,
    dst_crs: rio.CRS,
    dst_bounds: rio.coords.BoundingBox,
    dst_res: tuple[float, float],
    ) -> xdem.DEM:
    out_shape = int((dst_bounds.top - dst_bounds.bottom) / dst_res[1]), int((dst_bounds.right - dst_bounds.left) / dst_res[0])
    out_transform = rio.transform.from_bounds(*dst_bounds, *out_shape[::-1])

    gdal.UseExceptions()

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_vrt = Path(temp_dir) / "temp.vrt"

        variete.vrt.vrt.vrt_warp(
            temp_vrt,
            filepath,
            dst_crs=dst_crs,
            dst_shape=out_shape,
            dst_transform=out_transform,
        )

        return xdem.DEM(str(temp_vrt), load_data=True)

    
    
