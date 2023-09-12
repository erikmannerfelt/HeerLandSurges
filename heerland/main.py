import geopandas as gpd
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numba
import variete
import rasterio as rio
import shapely
import geoutils as gu

with warnings.catch_warnings():
    warnings.simplefilter("ignore", numba.NumbaDeprecationWarning)
    import xdem


import heerland.rasters
import heerland.utilities

def get_dem_paths() -> dict[str, str | Path]:
    base_dir = Path("/media/storage/Erik/Data/NPI/DEMs/")

    dem_1990_vsi = "/vsizip/" + str(base_dir.joinpath("NP_S0_DTM20_199095_33.zip/NP_S0_DTM20_199095_33/S0_DTM20_199095_33.tif"))

    dem_2010 = base_dir / "S0_DTM5/NP_S0_DTM5.tif"

    return {
        "1990": dem_1990_vsi,
        "2010-mosaic": dem_2010,
    }

def get_observed_surges(filepath: Path = Path("GIS/shapes/observed_surges.geojson"), this_year: int = 2023) -> gpd.GeoDataFrame:

    data = gpd.read_file(filepath).dropna(how="all", subset=["start_date", "end_date"])

    for date_col in ["start_date", "end_date"]:
        data.loc[data[date_col].str.len() == 0, date_col] = np.nan
        data[date_col] = data[date_col].dropna().str.split("-").apply(lambda arr: sum(float(value) / (60 ** i) for i, value in enumerate(arr)))

    # This is rough, but assume that any surge without an end observation ends 10 years later.
    data.loc[data["end_date"].isna(), "end_date"] = data["start_date"] + 10

    return data


def load_rgi(bounds: rio.coords.BoundingBox | None = None, crs: rio.CRS = rio.CRS.from_epsg(4326),  filepath = "zip://input/07_rgi60_Svalbard.zip/07_rgi60_Svalbard/region_07_rgi60_Svalbard_update.shp"):

    bounds_series = gpd.GeoSeries([shapely.geometry.box(*bounds)], crs=crs)

    return gpd.read_file(filepath, bbox=bounds_series).to_crs(crs)

    


def prepare_1990_2010_dems() -> dict[str, Path]:
    crs = rio.CRS.from_epsg(32633)
    res = (20.0,) * 2
    dem_paths = get_dem_paths()

    cache_dir = Path("cache/").absolute()
    cache_label = "prepare_1990_2010_dems-"

    cache_paths = {
        "1990": cache_dir / f"{cache_label}1990.tif",
        "2010-mosaic": cache_dir / f"{cache_label}2010.tif",
        "dhdt-1990-2010": cache_dir / f"{cache_label}dhdt.tif",
    }

    if all(p.is_file() for p in cache_paths.values()):
        return cache_paths

    boundary = gpd.read_file("GIS/shapes/study_area_outline.geojson")

    bounds = heerland.utilities.align_bounds(rio.coords.BoundingBox(*boundary.total_bounds), res, half_mod=False)

    rgi = load_rgi(bounds=bounds, crs=crs)

    dem_1990 = heerland.rasters.load_raster_subset(dem_paths["1990"], dst_crs=crs, dst_bounds=bounds, dst_res=res)
    # The ocean is 0.52 m for some reason
    dem_1990.data.mask[dem_1990.data == 0.52] = True

    inlier_mask = ~gu.Vector(rgi).create_mask(dem_1990)
    dem_2010 = heerland.rasters.load_raster_subset(dem_paths["2010-mosaic"], dst_crs=crs, dst_bounds=bounds, dst_res=res)
    dem_2010.data.mask[dem_2010.data == 0.] = True

    coreg = xdem.coreg.NuthKaab() + xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), subdivision=8) + xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), subdivision=64)

    coreg.fit(dem_2010, dem_1990, inlier_mask=inlier_mask)

    dem_1990_coreg = coreg.apply(dem_1990)
    dhdt_map = (dem_2010 - dem_1990_coreg) / (2010 - 1990)

    cache_paths["1990"].parent.mkdir(exist_ok=True, parents=True)
    dem_1990.save(cache_paths["1990"])
    dem_2010.save(cache_paths["2010-mosaic"])
    dhdt_map.save(cache_paths["dhdt-1990-2010"])

    return cache_paths
