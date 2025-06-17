import geopandas as gpd
import pandas as pd
from shapely import Point

file_path = r"data/external/gadm_410.gpkg"

# import necessary columns
gdf = gpd.read_file(
    file_path, columns=["GID_2", "NAME_0", "NAME_1", "NAME_2", "geometry"]
)

gdf = gdf.rename(
    columns={
        "GID_2": "subregion_id",
        "NAME_0": "country",
        "NAME_1": "region",
        "NAME_2": "subregion",
    }
)

# load csv with image id and coords
images_df = pd.read_csv("data/external/osv5m/test.csv")

# removing unecessary columns
drop_cols = [
    "creator_username",
    "creator_id",
    "sequence",
    "road_index",
    "lon_bin",
    "lat_bin",
    "cell",
    "quadtree_10_1000",
    "country",
    "region",
    "sub-region",
    "city",
    "unique_country",
    "unique_region",
    "unique_sub-region",
    "unique_city",
    "captured_at",
]

images_df = images_df.drop(columns=drop_cols)

# convert to geodataframe with point geometry
images_df["geometry"] = images_df.apply(
    lambda row: Point(row["longitude"], row["latitude"]), axis=1
)

images_gdf = gpd.GeoDataFrame(images_df, geometry="geometry", crs="EPSG:4326")

# spatial join using spatial index (automatically uses rtree or pygeos if installed)
labeled = gpd.sjoin(images_gdf, gdf, how="left", predicate="within")
labeled = labeled.drop(columns="index_right")
labeled.to_csv("data/processed/labeledtest.csv", index=False)
print(labeled[["subregion_id"]].head(10))
# print(gdf.head())
