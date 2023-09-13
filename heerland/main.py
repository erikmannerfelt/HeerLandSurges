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
import scipy.interpolate
import glacier_lengths.core

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

    bounds = heerland.utilities.get_study_bounds(res_mod=res[0], crs=crs) 

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


def _rec_get_tributary_centerline(line: pd.Series, lines: gpd.GeoDataFrame, rec: int, max_rec: int = 20):

    if line["outflow_id"] == -1 or rec > max_rec:
        return line[["rgi_id", "rgi_g_id", "segment_id", "geometry"]]

    parent_line = _rec_get_tributary_centerline(lines[lines["segment_id"] == line["outflow_id"]].iloc[0], lines=lines, rec= rec +1, max_rec=max_rec)

    line_coords = list(line.geometry.coords)
    line_coords.insert(-1, glacier_lengths.core._extrapolate_point(line_coords[-2], line_coords[-1]))

    extended_line = extrapolate_line(line.geometry, line.geometry.length * 0.5) 

    split = shapely.ops.split(parent_line.geometry, extended_line)

    lower = [part for part in split.geoms if part.touches(parent_line.geometry.interpolate(1, normalized=True))][0]

    combined = shapely.geometry.LineString(list(line.geometry.coords) + list(lower.coords))

    new_line = line.copy()
    new_line.geometry = combined

    return new_line[["rgi_id", "rgi_g_id", "segment_id", "geometry"]]




def cartesian_to_polar(x_coord: float, y_coord: float) -> tuple[float, float]:

    if x_coord == 0.:
        if y_coord > 0.:
            alpha = 0.
        else:
            alpha = np.pi
    else:
        alpha = np.arctan(np.abs(y_coord / x_coord))

        if x_coord < 0 and y_coord >= 0:
            alpha += np.pi * 1.5
        elif x_coord < 0 and y_coord < 0:
            alpha += np.pi
        elif x_coord > 0 and y_coord < 0:
            alpha += np.pi * 0.5
            

    radius = np.sqrt(np.sum(np.square([x_coord, y_coord])))

    return alpha, radius


def test_cartesian_to_polar():


    x_coord = 1
    y_coord = 1

    alpha, radius = cartesian_to_polar(x_coord, y_coord)
    assert alpha == np.pi / 4
    assert radius == 2 ** 0.5

    x_coord = -1
    alpha, radius = cartesian_to_polar(x_coord, y_coord)
    assert alpha == np.pi * 1.75
    assert radius == 2 ** 0.5

    y_coord = -1
    alpha, radius = cartesian_to_polar(x_coord, y_coord)
    assert alpha == np.pi * 1.25

    x_coord = 1
    alpha, radius = cartesian_to_polar(x_coord, y_coord)
    assert alpha == np.pi * 0.75
    

def extrapolate_line(line: shapely.geometry.LineString, length: float, backward: bool = False) -> shapely.geometry.LineString:

    coords = list(line.coords)

    if backward:
        before_pt = coords[1]
        after_pt = coords[0]
    else:
        before_pt = coords[-2]
        after_pt = coords[-1]

    x_diff = after_pt[0] - before_pt[0]
    y_diff = after_pt[1] - before_pt[1]

    alpha, _ = cartesian_to_polar(x_diff, y_diff)

    ext_xdiff = length * np.sin(alpha)
    ext_ydiff = length * np.cos(alpha)

    coords.insert(0 if backward else -1, [after_pt[0] + ext_xdiff, after_pt[1] + ext_ydiff])

    return shapely.geometry.LineString(coords)

    
    
    
    

def get_tributary_centerline(lines: gpd.GeoDataFrame):

    full_lines = []

    for _, line in lines.sort_values("strahler_n", ascending=False).iterrows():
        full_lines.append(_rec_get_tributary_centerline(line, lines, rec=0).to_dict())

    full_lines = gpd.GeoDataFrame.from_records(full_lines)
    full_lines.crs = lines.crs
    return full_lines


def make_tributary_masks(rgi_id: str, bounds: rio.coords.BoundingBox, crs: rio.CRS, res: tuple[float, float]):
    dem_paths = heerland.main.prepare_1990_2010_dems()

    rgi = heerland.inputs.rgi.read_rgi7(bounds=bounds, crs=crs)
    centerlines = heerland.inputs.rgi.read_rgi7_centerlines(bounds=bounds, crs=crs)

    outline_df = rgi.query(f"rgi_id == '{rgi_id}'")
    outline = outline_df.iloc[0]

    dem = xdem.DEM(dem_paths["2010-mosaic"], load_data=False)
    outline_bounds = heerland.utilities.align_bounds(rio.coords.BoundingBox(*outline.geometry.bounds), res=dem.res, half_mod=False)
    dem.crop(list(outline_bounds), inplace=True)
    outline_mask = gu.Vector(outline_df).create_mask(dem).data

    x_coords, y_coords = np.meshgrid(
        np.linspace(dem.bounds.left + dem.res[0] / 2, dem.bounds.right - dem.res[0] / 2, dem.shape[1]),
        np.linspace(dem.bounds.bottom + dem.res[1] / 2, dem.bounds.top - dem.res[1] / 2, dem.shape[0])[::-1]
    )
    centerlines = centerlines.query(f"rgi_g_id == '{outline['rgi_id']}'").sort_values("strahler_n")

    tributary_lines = get_tributary_centerline(centerlines)
    tributary_masks = {}
    for _, tributary in tributary_lines.iterrows():

        lines = centerlines.query(f"segment_id != {tributary.segment_id}").copy()
        lines.geometry = centerlines.difference(tributary.geometry)

        lines.loc[lines.index.max() + 1] = tributary

        points = []
        for _, line in lines.iterrows():

            for distance in np.arange(0, line.geometry.length, step=dem.res[0]):
                point = line.geometry.interpolate(distance)

                points.append([point.x, point.y, line["segment_id"]])
             

        points = np.array(points)

        mesh = scipy.interpolate.griddata(
            points[:, :2],
            points[:, 2],
            xi=(x_coords, y_coords),
            method="nearest",
            fill_value=-1,
        )
        mesh[~outline_mask] = -1

        tributary_masks[tributary["rgi_id"]] = mesh == tributary["segment_id"]

    return tributary_masks
