import os
import glob
import pandas as pd
import geopandas as gpd
import warnings
import sys

# Ignore common warnings from geopandas
warnings.filterwarnings('ignore')

# --- Configuration ---

# Define the target Coordinate Reference System (CRS) for all operations
TARGET_CRS = "epsg:3571"

# Get year from command line and convert to integer
# if len(sys.argv) != 2:
#     print("Usage: python script.py <year>")
#     sys.exit(1)

# year = int(sys.argv[1])\


year = 2025

# Base output path where subfolders for each region will be created
base_out_path = '/explore/nobackup/people/spotter5/for_reporters/viirs_by_day_multiple_aoi'
os.makedirs(base_out_path, exist_ok=True)

# --- Load and Prepare all AOI Shapefiles ---
print("Loading and preparing Area of Interest (AOI) shapefiles...")
shapefile_info = [
    ('circumpolar', '/explore/nobackup/people/spotter5/viirs_nrt/shared_data/shapefiles/circumpolar.gpkg'),
    ('siberia', '/explore/nobackup/people/spotter5/arctic_report_card/shapes/Siberia/Siberia_Domain_V1_dissolved.shp'),
    ('alaska', '/explore/nobackup/people/spotter5/arctic_report_card/shapes/ak_permafrost.shp'),
    ('canada', '/explore/nobackup/people/spotter5/arctic_report_card/shapes/ca_permafrost_bor_tun.shp'),
]

prepared_aois = []
for name, path in shapefile_info:
    try:
        aoi_gdf = gpd.read_file(path).to_crs(TARGET_CRS)
        aoi_gdf['geometry'] = aoi_gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
        prepared_aois.append((name, aoi_gdf))
        print(f"  - Successfully loaded and prepared '{name}'")
    except Exception as e:
        print(f"  - Failed to load AOI '{name}' from {path}. Error: {e}")

# --- Main Processing ---

print(f"\n--- Processing Year: {year} ---")

# 1. Determine input paths and gather all files for the year
all_files = []
if year >= 2024:
    print(f"Using new paths for {year} and concatenating sources.")
    in_path_ak_ca = f"/explore/nobackup/people/spotter5/viirs_nrt/2_pipeline/ak_ca/{year}/Snapshot"
    in_path_ea = f"/explore/nobackup/people/spotter5/viirs_nrt/2_pipeline/ea/{year}/Snapshot"
    all_files.extend(glob.glob(os.path.join(in_path_ak_ca, '*.gpkg')))
    all_files.extend(glob.glob(os.path.join(in_path_ea, '*.gpkg')))
else:
    in_path_historical = f"/explore/nobackup/people/spotter5/viirs_nrt/rebecca_historical/fire_atlas/fire_atlas/{year}/Snapshot"
    all_files = glob.glob(os.path.join(in_path_historical, '*.gpkg'))

# Filter out files containing "_FL" or "_NFP"
filtered_files = [f for f in all_files if not ("_FL" in f or "_NFP" in f)]

if not filtered_files:
    print(f"No valid data files found for {year}. Skipping.")
    sys.exit(0)

print(f"Found {len(filtered_files)} files to process for {year}.")

# 2. Read and concatenate all fire data for the year
yearly_fire_data = []
for f in filtered_files:
    try:
        in_df = gpd.read_file(f)[['fireid', 'farea', 'tst_year', 'ted_doy', 'geometry']]
        yearly_fire_data.append(in_df)
    except Exception as e:
        print(f"Error processing file {f}: {e}")

# Proceed only if data was successfully read
if yearly_fire_data:
    full_year_gdf = pd.concat(yearly_fire_data, ignore_index=True)
    full_year_gdf = gpd.GeoDataFrame(full_year_gdf, geometry='geometry')
    full_year_gdf.set_crs(gpd.read_file(filtered_files[0]).crs, inplace=True)

    # 3. Process the combined yearly data
    print("Fixing geometries and reprojecting yearly data...")
    full_year_gdf['geometry'] = full_year_gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    full_year_gdf = full_year_gdf.to_crs(TARGET_CRS)

    # 4. Loop through each prepared AOI, clip, and save
    for aoi_name, aoi_gdf in prepared_aois:
        print(f"  Clipping to '{aoi_name}'...")

        aoi_out_path = os.path.join(base_out_path, aoi_name)
        os.makedirs(aoi_out_path, exist_ok=True)

        clipped_gdf = gpd.clip(full_year_gdf, aoi_gdf)

        if not clipped_gdf.empty:
            clipped_gdf['farea'] = (clipped_gdf['farea'] * 100) / 1e6
            out_filepath = os.path.join(aoi_out_path, f"{year}.shp")
            clipped_gdf.to_file(out_filepath)
            print(f"    ✅ Saved output to {out_filepath}")
        else:
            print(f"    - No data for '{aoi_name}' in {year} after clipping.")

print("\n🎉 Processing complete.")
