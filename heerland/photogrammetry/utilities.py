import geopandas as gpd
from pathlib import Path
import rasterio as rio
import numpy as np
import warnings

def check_image_availability(pgm_input_dir: Path = Path("photogrammetry/input/"), aerial_image_loc_path: Path = Path("GIS/shapes/aerial_images.geojson")):

    image_meta = gpd.read_file(aerial_image_loc_path)

    # temporary
    #image_meta = image_meta.query("survey == 'S61'")

    missing: dict[str, int | list[str]] = {}

    for survey_id, survey_data in image_meta.groupby("survey"):

        image_dir = pgm_input_dir.joinpath(survey_id)

        if not image_dir.is_dir():
            missing[survey_id] = survey_data.shape[0]
            continue

        missing[survey_id] = []

        image_names = survey_data["survey"] + "_" + survey_data["image_id"] + ".tif"

        for _, filename in image_names.items():

            filepath = image_dir / filename

            if not filepath.is_file():
                missing[survey_id].append(filename)

    if len(missing) == 0:
        return

    n_total_missing = 0
    for survey_id in missing:
        if isinstance(missing[survey_id], int):
            print(f"{survey_id} is missing {missing[survey_id]} images (no directory)")
            n_total_missing += missing[survey_id]
            continue
            
        if (n_missing := len(missing[survey_id])) == 0:
            print(f"{survey_id} is complete")
            continue

        print(f"{survey_id} is missing {n_missing} images")

        n_total_missing += n_missing

    print(f"Missing {n_total_missing} images in total")


def generate_gcp_file(
    gcp_filepath: Path = Path("GIS/shapes/gcps.geojson"),
    dem_filepath: Path = Path("cache/prepare_1990_2010_dems-2010.tif"),
    output_filepath: Path = Path("photogrammetry/gcps.csv")
    ):

    gcps = gpd.read_file(gcp_filepath)
    gcps["easting"] = gcps.geometry.x
    gcps["northing"] = gcps.geometry.y

    
    with rio.open(dem_filepath) as raster, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*converting a masked element.*")

        gcps["elevation"] = np.fromiter(raster.sample(gcps[["easting", "northing"]].values, masked=True), dtype=raster.dtypes[0], count=gcps.shape[0])


    gcps[["name", "easting", "northing", "elevation"]].dropna().to_csv(output_filepath, index=False)

    

