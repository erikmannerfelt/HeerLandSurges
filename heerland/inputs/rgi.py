import geopandas as gpd
import rasterio as rio
import shapely

def read_rgi7(bounds: rio.coords.BoundingBox, crs: rio.CRS, filepath: str = "zip://input/RGI7/RGI2000-v7.0-G-07_svalbard_jan_mayen.zip/RGI2000-v7.0-G-07_svalbard_jan_mayen.shp"):

    bounds_series = gpd.GeoSeries([shapely.geometry.box(*bounds)], crs=crs)

    return gpd.read_file(filepath, bbox=bounds_series).to_crs(crs)
    

def read_rgi7_centerlines(bounds: rio.coords.BoundingBox, crs: rio.CRS, filepath: str = "zip://input/RGI7/RGI2000-v7.0-L-07_svalbard_jan_mayen.zip/RGI2000-v7.0-L-07_svalbard_jan_mayen.shp"):

    bounds_series = gpd.GeoSeries([shapely.geometry.box(*bounds)], crs=crs)

    return gpd.read_file(filepath, bbox=bounds_series).to_crs(crs)
