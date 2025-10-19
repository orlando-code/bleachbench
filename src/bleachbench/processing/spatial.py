import geopandas as gpd
import pyproj

def calc_true_buffer(distance: float, minx: float, miny: float, maxy: float) -> float:
    """Calculate the equal-area buffer value from a distance in degrees"""
    center_lat = (miny + maxy) / 2
    # determine length in meters for 'distance' degrees at center latitude
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist_lat = geod.inv(minx, center_lat, minx, center_lat + distance)
    return dist_lat # latitude value is safer near equator


def buffer_shapes(df: gpd.GeoDataFrame, buffer: float) -> gpd.GeoDataFrame:
    """Buffer shapes in equal-area projection, then cast back to geographic CRS"""
    return df.to_crs("EPSG:32662").buffer(buffer).to_crs("EPSG:4326")
    
    
def points_in_polygons(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Find points that are within polygons"""
    joined = gpd.sjoin(
        points,
        gpd.GeoDataFrame(polygons, columns=["geometry"], crs="EPSG:4326") if not isinstance(polygons, gpd.GeoDataFrame) else polygons, 
        how="inner",
    )
    # drop duplicates
    joined.reset_index(drop=False, inplace=True)
    return joined.drop_duplicates(subset="index")