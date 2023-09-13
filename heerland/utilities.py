import rasterio as rio
import geopandas as gpd

def align_bounds(
    bounds: rio.coords.BoundingBox | dict[str, float],
    res: tuple[float, float] | None = None,
    half_mod: bool = True,
    buffer: float | None = None,
) -> rio.coords.BoundingBox:
    if isinstance(bounds, rio.coords.BoundingBox):
        bounds = {key: getattr(bounds, key) for key in ["left", "bottom", "right", "top"]}

    # Ensure that the moduli of the bounds are zero
    for i, bound0 in enumerate([["left", "right"], ["bottom", "top"]]):
        for j, bound in enumerate(bound0):

            mod = (bounds[bound] - (res[i] / 2 if half_mod else 0)) % res[i]

            bounds[bound] = (
                bounds[bound] - mod + (res[i] if i > 0 and mod != 0 else 0) + ((buffer or 0) * (1 if i > 0 else -1))
            )
    return rio.coords.BoundingBox(**bounds)


def get_study_bounds(
    res_mod: float = 20.,
    crs: rio.CRS | None = None,
    ) -> rio.coords.BoundingBox:
    boundary = gpd.read_file("GIS/shapes/study_area_outline.geojson")

    if crs is not None:
        boundary = boundary.to_crs(crs)

    return align_bounds(rio.coords.BoundingBox(*boundary.total_bounds), (res_mod,) * 2, half_mod=False)
