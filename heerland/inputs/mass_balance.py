import heerland.utilities
from heerland.constants import CONSTANTS
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import pyproj
import numpy as np
import scipy.optimize
import scipy.interpolate
from pathlib import Path
import rasterio as rio
import pandas as pd
import sklearn
import sklearn.pipeline
import sklearn.linear_model
from tqdm import tqdm


def get_schmidt_cmb(
    url: str = "https://thredds.met.no/thredds/fileServer/arcticdata/nl/CARRASvalbard/Monthly_means/Svalbard_CARRA_cmb_1991_2021_MM.nc",
    dem_path: Path = Path("cache/prepare_1990_2010_dems-2010.tif"),
    polynomial_degree: int = 3,
) -> Path:
    """
    Download and downscale the CMB of Schmidt et al., (2023).

    The data are downscaled by assuming a polynomial relationship between x, y and  (DEM) elevation for each timestep.

    Parameters
    ----------
    - url: The URL for the CMB dataset
    - dem_path: The path to the DEM to use for downscaling
    - polynomial_degree: The degree of the polynomial to use for the downscaling model.

    Returns
    -------
    The path to the downscaled dataset.
    """
    output_path = Path("cache/").joinpath("schmidt_cmb.nc")

    if output_path.is_file():
        return output_path
    filepath = heerland.utilities.download_large_file(url=url, directory=CONSTANTS.data_dir)

    # Load the DEM and generate x/y coords to interpolate later
    with rio.open(dem_path) as raster:
        x_coords, y_coords = np.meshgrid(
            np.linspace(raster.bounds.left + raster.res[0] / 2, raster.bounds.right - raster.res[0] / 2, num=raster.width),
            np.linspace(raster.bounds.bottom + raster.res[1] / 2, raster.bounds.top - raster.res[1] / 2, num=raster.height)[::-1]
        )
        dem_arr = raster.read(1, masked=True).filled(0)
        crs = raster.crs
        bounds = raster.bounds

    with xr.open_dataset(filepath) as data:

        #data = data.isel(time=slice(2))

        # Save metadata that will be propagated to the output file
        times = data.time.values
        cmb_attrs = data["cmb"].attrs
        # Initialize the downscaled CMB array
        cmbs = np.empty((times.size,) + dem_arr.shape, dtype="float32")

        # Convert the weirdly projected raster to a point cloud.
        data = data.stack(xy=["x", "y"])
        data = data.drop_vars(['x', 'xy', 'y']).assign_coords(xy=np.arange(data.xy.shape[0]))

        # Project the WGS84 coordinates to the destination coords.
        easting, northing = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs, always_xy=True).transform(data["lon"], data["lat"])
        data["easting"] = "xy", easting
        data["northing"] = "xy", northing

        # Filter out the points to only have valid points within the bounding box.
        data = data.where(
            (data["easting"] >= bounds.left) &
            (data["easting"] <= bounds.right) &
            (data["northing"] >= bounds.bottom) &
            (data["northing"] <= bounds.top) &
            (~data["cmb"].isnull()),
            drop=True
        )

        # This gets broadcast weirdly to all times
        for variable in ["easting", "northing", "gmask"]:
            data[variable] = data[variable].isel(time=0)

        # Grid the index of each point using nearest neighbor to sample the DEM well.
        # Basically, we want the mean of all DEM elevations within the cell size of the CMB data.
        # TODO: Mask this with a maximum distance, e.g. 2.5 km * sqrt(2)
        gridded_xy = scipy.interpolate.griddata(
            (data["easting"], data["northing"]),
            data["xy"],
            xi=(x_coords, y_coords),
            method="nearest",
            fill_value=0
        )

        # Find the mean elevation for each point index
        gridded = pd.DataFrame({"xy": gridded_xy.ravel(), "z": dem_arr.ravel()}).groupby("xy").mean()
        # Assign the mean elevation to each point.
        data["elevation"] = "xy", gridded.loc[data["xy"].values].values.ravel()

        # Initialize a polynomial to model the CMB in space
        model = sklearn.pipeline.make_pipeline(sklearn.preprocessing.PolynomialFeatures(degree=polynomial_degree), sklearn.linear_model.LinearRegression(fit_intercept=True))

        # Convert the data to a dataframe as it was hard to work this with xarray
        data_df = data[["easting", "northing", "elevation"]].to_pandas()

        # For each timestep, estimate a polynomial relationship and predict it (downscale)
        for i, time in tqdm(enumerate(times), total=times.size, desc="Downscaling data."):
            data_df["cmb"] = data["cmb"].sel(time=time).values

            model.fit(data_df[["easting", "northing", "elevation"]].values, data_df["cmb"].values)

            cmbs[i, :, :] = model.predict(np.vstack((x_coords.ravel(), y_coords.ravel(), dem_arr.ravel())).T).reshape(dem_arr.shape)


    # Create a Dataset from the data.
    cmbs = xr.DataArray(cmbs, coords=[("time", times), ("y", y_coords[:, 0]),("x", x_coords[0, :])], name="cmb", attrs=cmb_attrs).to_dataset()

    # Assign some spatial metadata
    cmbs.rio.write_crs(crs, inplace=True, grid_mapping_name="grid_mapping")
    cmbs.rio.write_transform(grid_mapping_name="grid_mapping", inplace=True)


    cmbs.to_netcdf(output_path, encoding={variable: {"zlib": True, "complevel": 5} for variable in cmbs.data_vars}, engine="netcdf4")

    return output_path
