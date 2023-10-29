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
import skimage.segmentation
import skimage.morphology
import skimage.measure
import shapely.geometry
import scipy.spatial
import scipy.interpolate
import pandas as pd
import xarray as xr

with warnings.catch_warnings():
    warnings.simplefilter("ignore", numba.NumbaDeprecationWarning)
    import xdem

import heerland.main
import heerland.utilities
import heerland.rasters
import heerland.inputs.rgi
import heerland.inputs.mass_balance


@numba.njit(parallel=False)
def _catchment_nb(dem: np.ndarray, width: int, height: int, point: tuple[int, int], current_mask: np.ndarray, neighbor_template: np.ndarray, rec: int = 0, max_rec: int = 20) -> np.ndarray:

    if rec > max_rec:
        return current_mask

    value = dem[point[0] * width + point[1]]

    neighbors = neighbor_template.copy()
    neighbors[:, 0] += point[0]
    neighbors[:, 1] += point[1]

    higher = dem[neighbors[:, 0] * width + neighbors[:, 1]] > value

    neighbors = neighbors[higher.astype("bool")]

    current_mask[neighbors[:, 0] * width + neighbors[:, 1]] = 1

    for neighbor in neighbors:

        new_mask = _catchment_nb(dem, point=neighbor, height=height, width=width, current_mask=current_mask, neighbor_template=neighbor_template, rec=rec + 1, max_rec=max_rec)
        current_mask += new_mask - current_mask


    return current_mask
            

def make_catchment(dem: np.ndarray, point: tuple[int, int], max_rec: int = 6) -> np.ndarray:
    mask = np.zeros(dem.shape, dtype="int64").ravel()
    neighbors = np.dstack(np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))).reshape(-1, 2)
    neighbors = neighbors[~np.all(neighbors == 0, axis=1)]

    height = dem.shape[0]
    width = dem.shape[1]
    mask[point[0] * width + point[1]] = 1

    assert dem[point[0], point[1]] == dem.ravel()[point[0] * width + point[1]]

    return _catchment_nb(dem.ravel(), height=height, width=width, point=point, current_mask=mask, max_rec=max_rec, neighbor_template=neighbors).reshape((height, width))


@numba.njit()
def _get_neigbors(point, height, width):

    neighbors = [0]
    neighbors.clear()

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue

            new_point = point + width * i + j

            if new_point < 0 or new_point > (width * height):
                continue
            neighbors.append(new_point) 

    return neighbors
    

@numba.njit(parallel=False)
def _catchment_nb2(dem, width, height, point, initial_mask: np.ndarray | None = None, max_iters: int = 10000):
    finished = [0]
    finished.clear()

    if initial_mask is not None:
        mask = initial_mask

        # mask2d = mask.reshape((height, width))

        # diff_row = np.abs(mask2d[:-1, :] - mask2d[1:, :])

        # new_row = np.zeros(mask2d.shape[1])[None, :]
        # diff_row = np.concatenate((new_row, diff_row), axis=0)

        # diff_col = np.abs(mask2d[:, :-1] - mask2d[:, 1:,])
        # new_col = np.zeros(mask2d.shape[0])[:, None]
        # diff_col = np.concatenate((new_col,diff_col), axis=1) 

        # diffs = np.where(diff_row > diff_col, diff_row, diff_col)

        # rim = [0]
        # rim.clear()
        # for i in np.argwhere(diffs.ravel() > 0).ravel():
        #     rim += _get_neigbors(i, height=height, width=width)

        
        # rim += _get_neigbors(point, height=height, width=width)

        for i in np.argwhere(mask > 0).ravel():
            finished.append(i)
    else:
        mask = np.zeros(dem.shape, dtype="uint8")


    rim = _get_neigbors(point, height=height, width=width)

    for _ in range(max_iters):

        new_rim = [0]
        new_rim.clear()

        for rim_point in rim:

            for neighbor in _get_neigbors(rim_point, height=height, width=width):
                if neighbor in rim:
                    continue

                if neighbor in finished:
                    continue

                if dem[neighbor] < dem[rim_point]:
                    continue

                mask[neighbor] = 1

                finished.append(neighbor)

                new_rim.append(neighbor)


        if len(rim) == 0:
            break

        rim = new_rim

        # rim.append(point)

        # rim = np.array(rim)
        # rim_rows = rim % width

        # rim_cols = ((rim - rim_rows) / width).astype(int)

        # plt.imshow(dem.reshape(height, width))
        # plt.scatter(rim_rows, rim_cols)
        # plt.show()

               
    return mask


def make_catchment2(dem: np.ndarray, point: tuple[int, int], initial_mask: np.ndarray | None = None) -> np.ndarray:

    height = dem.shape[0]
    width = dem.shape[1]
    point = point[0] * width + point[1]

    if initial_mask is not None:
        initial_mask = initial_mask.ravel()

    return _catchment_nb2(dem=dem.ravel(), height=height, width=width, point=point, initial_mask=initial_mask).reshape(height, width)


def fill_mask_holes(mask: np.ndarray):
    mask = mask.astype(bool)

    labelled = skimage.measure.label(~mask)

    values, counts = np.unique(labelled, return_counts=True)

    counts[values == 0] = 0

    mask = labelled != values[np.argmax(counts)]

    return mask

def extract_largest_mask_feature(mask: np.ndarray):
    mask = mask.astype(bool)

    labelled = skimage.measure.label(mask)
    values, counts = np.unique(labelled, return_counts=True)

    counts[values == 0] = 0

    return labelled == values[np.argmax(counts)]
    


def make_centerline_catchment(dem: xdem.DEM, line: shapely.geometry.LineString):
    mask = None
    dem_arr = dem.data.filled(0)

    for val in range(-10, -2):
        
        ij = [round(v[0]) for v in dem.xy2ij(*line.coords[val])]

        marker_arr = np.zeros(dem.data.shape, dtype=int)

        marker_arr[ij[0], ij[1]] = 1

        catchment = make_catchment2(dem_arr, tuple(ij), initial_mask=mask.copy() if mask is not None else None)

        if mask is not None:
            print(np.count_nonzero(catchment), np.count_nonzero(catchment) / np.count_nonzero(mask))
        if mask is not None and (np.count_nonzero(catchment) / np.count_nonzero(mask)) > 1.5:
            break

        mask = catchment

    return extract_largest_mask_feature(fill_mask_holes(mask))


def get_edvard_cmb():
    cmb_path = heerland.inputs.mass_balance.get_schmidt_cmb()

    poi = [557181, 8642986]
    buffer = 200
    with xr.open_dataset(cmb_path, chunks="auto") as cmb:
        cmb = cmb.sel(x=slice(poi[0] - buffer, poi[0] + buffer), y=slice(poi[1] + buffer, poi[1] - buffer)).mean(["y", "x"])["cmb"] / 1000

        cmb.cumsum().plot()
        plt.show()


def get_ragnamarie_cmb(study_bounds, crs):
    
    cmb = xr.open_dataset(heerland.inputs.mass_balance.get_schmidt_cmb())

    with rio.open("cache/prepare_1990_2010_dems-2010.tif") as raster:
        cmb["ref_dem"] = ("y", "x"), raster.read(1, masked=True).filled(0)



    rgi = heerland.inputs.rgi.read_rgi7(bounds=study_bounds, crs=crs).query("glac_name == 'Ragna-Mariebreen'")

    ragna_bounds = rgi.total_bounds
    outline = rgi.geometry.iloc[0]

    cmb = cmb.sel(x=slice(ragna_bounds[0], ragna_bounds[2]), y=slice(ragna_bounds[3], ragna_bounds[1]))

    cmb = cmb.stack(xy=["x", "y"])

    cmb["mask"] = ("xy"), ([outline.contains(shapely.geometry.Point(*xy)) for xy in cmb["xy"].values])  

    cmb = cmb.where(cmb["mask"], drop=True).swap_dims(xy="ref_dem").reset_coords(drop=True)

    bins = np.arange(0, cmb["ref_dem"].values.max() + 50, step=50)

    cmb_gradient = cmb["cmb"].groupby((cmb["ref_dem"] / 50).astype(int)).mean()

    cmb_gradient["ref_dem"] = bins[cmb_gradient["ref_dem"].values] + np.mean(np.diff(bins)) / 2 

    gradient_df = cmb_gradient.to_dataframe().squeeze().unstack(level=1)
    gradient_df.columns = gradient_df.columns.astype(int)

    gradient_df.to_csv("ragnamarie_cmb_1991-2021.csv")
    mean_gradient = gradient_df.mean(axis="index")
    mean_gradient.index.name = "elevation"
    mean_gradient.name = "cmb"
    mean_gradient.to_csv("ragnamarie_mean_cmb.csv")
        

def main():
    crs = rio.CRS.from_epsg(32633)
    study_bounds = heerland.utilities.get_study_bounds(crs=crs)


    return

    ids = {
        "skobreen": "RGI2000-v7.0-G-07-00746",
        "vallakra": "RGI2000-v7.0-G-07-01538",
    }

    rgi_id = ids["vallakra"]
     
    study_bounds = heerland.utilities.get_study_bounds(crs=crs)

    masks = heerland.main.make_tributary_masks(rgi_id=rgi_id, bounds=study_bounds, crs=crs, res=[20., 20.])

    for i, (key, value) in enumerate(masks.items(), start=1):
        plt.subplot(1, len(masks), i)
        plt.imshow(value)

    plt.show()


    





if __name__ == "__main__":
    main()
