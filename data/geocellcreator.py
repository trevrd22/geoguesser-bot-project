import geopandas as gpd

file_path = r"D:\ML_Projects\geoguesser-bot-project\data\external\gadm_410.gpkg"

gdf = gpd.read_file(file_path)


print(gdf.columns)
print(gdf.head())