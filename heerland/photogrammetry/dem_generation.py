import os
import json
import subprocess
from pathlib import Path
import rasterio as rio
import tempfile
import warnings


def pc_to_dh(
    point_cloud_path: Path,
    output_dh_path: Path,
    dem_path: Path,
    ):
    point_cloud_path = Path(point_cloud_path)
    output_dh_path = Path(output_dh_path)
    dem_path = Path(dem_path)

    with rio.open(dem_path) as raster:
        bounds = raster.bounds
        resolution = raster.res[0]
        crs_epsg = raster.crs.to_epsg()

    pipeline = [
        str(point_cloud_path.absolute()),
        {
            "type": "filters.hag_dem",
            "raster": str(dem_path.absolute()),
        },
        {
            "type": "filters.assign",
            "value": "HeightAboveGround = HeightAboveGround * -1",
        },
        {
            "resolution": str(resolution),
            "bounds": f"([{bounds.left}, {bounds.right}], [{bounds.bottom}, {bounds.top}])",
            "filename": str(output_dh_path),
            "output_type": "idw",
            "dimension": "HeightAboveGround",
            "data_type": "float32",
            "override_srs": f"EPSG:{crs_epsg}",
            "gdalopts": ["COMPRESS=DEFLATE", "PREDICTOR=3", "ZLEVEL=12", "TILED=YES"]
        }
    ]

    result = run_pdal_pipeline(json.dumps(pipeline))

    print(result)


def dh_s61():

    pc_to_dh(
        "photogrammetry/temp/s61_dense_20230916.laz",
        "cache/dh_1961-2010.tif",
        "cache/prepare_1990_2010_dems-2010.tif",
    )
        
def dh_s56():
    pc_to_dh(
        "photogrammetry/temp/s56_dense_20230916.laz",
        "cache/dh_1956-2010.tif",
        "cache/prepare_1990_2010_dems-2010.tif",
    )

def dh_s69():
    pc_to_dh(
        "photogrammetry/temp/s69_dense_20230916.laz",
        "cache/dh_1969-2010.tif",
        "cache/prepare_1990_2010_dems-2010.tif",
    )

def run_pdal_pipeline(pipeline: str, output_metadata_file: str | None = None,
                      parameters: dict[str, str] | None = None, show_warnings: bool = False) -> dict[str, object]:
    """
    Run a PDAL pipeline.

    :param pipeline: The pipeline to run.
    :param output_metadata_file: Optional. The filepath for the pipeline metadata.
    :param parameters: Optional. Parameters to fill the pipeline with, e.g. {"FILEPATH": "/path/to/file"}.
    :param show_warnings: Show the full stdout of the PDAL process.

    :returns: output_meta: The metadata produced by the output.
    """
    # Create a temporary directory to save the output metadata in
    temp_dir = tempfile.TemporaryDirectory()
    # Fill the pipeline with
    if parameters is not None:
        for key in parameters:
            # Warn if the key cannot be found in the pipeline
            if key not in pipeline:
                warnings.warn(
                    f"{key}:{parameters[key]} given to the PDAL pipeline but the key was not found", RuntimeWarning)
            # Replace every occurrence of the key inside the pipeline with its corresponding value
            pipeline = pipeline.replace(key, str(parameters[key]))

    try:
        json.loads(pipeline)  # Throws an error if the pipeline is poorly formatted
    except json.decoder.JSONDecodeError as exception:
        raise ValueError("Pipeline was poorly formatted: \n" + pipeline + "\n" + str(exception))

    # Run PDAL with the pipeline as the stdin
    commands = ["pdal", "pipeline", "--stdin", "--metadata", os.path.join(temp_dir.name, "meta.json")]
    result = subprocess.run(commands, input=pipeline, check=False, stdout=subprocess.PIPE,
                            encoding="utf-8", stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise ValueError(f"PDAL failed: {result.stderr}")
    stdout = result.stdout

    if show_warnings and len(stdout.strip()) != 0:
        warnings.warn(stdout)

    # Load the temporary metadata file
    with open(os.path.join(temp_dir.name, "meta.json")) as infile:
        output_meta = json.load(infile)

    # Save it with a different name if one was provided
    if output_metadata_file is not None:
        with open(output_metadata_file, "w") as outfile:
            json.dump(output_meta, outfile)

    return output_meta
