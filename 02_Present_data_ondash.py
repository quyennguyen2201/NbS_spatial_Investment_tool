import base64
from functools import lru_cache
import geopandas as gpd
import json
import numpy as np
import os
import pandas as pd
import rasterio
import struct
import zlib
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode

import dash
import dash_leaflet as dl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from dash_extensions.javascript import assign
from dash_leaflet import EditControl
from flask import Response, request
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.interpolate import interp1d
from shapely.geometry import shape
from shapely.ops import unary_union

def custom_aoi_to_gdf(geojson):
    if not geojson or not geojson.get("features"):
        return None

    geometries = [
        shape(feature["geometry"])
        for feature in geojson["features"]
    ]

    aoi_geometry = unary_union(geometries)

    return gpd.GeoDataFrame(
        {"name": ["Custom AOI"]},
        geometry=[aoi_geometry],
        crs="EPSG:4326",
    )


EMPTY_GEOJSON = {"type": "FeatureCollection", "features": []}


def _png_chunk(chunk_type, data):
    chunk = chunk_type + data
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)


@lru_cache(maxsize=4)
def raster_overlay_data_uri(scenario):
    raster_path = river_flood_scenario_paths[scenario]
    with rasterio.open(raster_path) as source:
        scale = min(1.0, 1400 / max(source.width, source.height))
        width = max(1, round(source.width * scale))
        height = max(1, round(source.height * scale))
        depths = source.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.average,
        )
        bounds = source.bounds

    valid = np.isfinite(depths) & (depths > 0)
    stops = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    palette = np.array([
        [204, 229, 255],
        [153, 204, 255],
        [102, 178, 255],
        [51, 153, 255],
        [0, 122, 204],
        [0, 76, 153],
    ])
    scaled = np.nan_to_num(np.clip(depths / 3.0, 0, 1), nan=0.0, posinf=1.0)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for channel in range(3):
        rgba[:, :, channel] = np.interp(scaled, stops, palette[:, channel]).astype(np.uint8)
    rgba[:, :, 3] = np.where(valid, 175, 0).astype(np.uint8)

    scanlines = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    raster_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    return data_uri, raster_bounds


def viewport_bbox(viewport):
    west, south, east, north = gdf_napier.total_bounds
    if not isinstance(viewport, dict):
        viewport = {
            "center": [center_lat, center_lon],
            "zoom": 16,
        }

    center = viewport.get("center")
    zoom = viewport.get("zoom", 12)
    if isinstance(center, dict):
        latitude = center.get("lat", center_lat)
        longitude = center.get("lng", center.get("lon", center_lon))
    elif isinstance(center, (list, tuple)) and len(center) >= 2:
        latitude, longitude = center[:2]
    else:
        return west, south, east, north

    zoom = max(float(zoom or 16), 1)
    longitude_span = 360 * 1000 / (256 * 2**zoom)
    latitude_span = longitude_span * np.cos(np.radians(latitude)) * 0.8
    return (
        longitude - longitude_span * 0.65,
        latitude - latitude_span * 0.65,
        longitude + longitude_span * 0.65,
        latitude + latitude_span * 0.65,
    )


def building_geojson_url(viewport, scenario, selected_layer):
    west, south, east, north = viewport_bbox(viewport)
    return "/api/building-footprints?" + urlencode({
        "west": round(west, 6),
        "south": round(south, 6),
        "east": round(east, 6),
        "north": round(north, 6),
        "scenario": scenario if scenario in {"0C", "1C", "2C", "3C"} else "1C",
        "measure": "cost" if scenario == "show replacement costs only" else "depth",
        "visible": selected_layer == "buildings",
        "zoom": (viewport.get("zoom") or 0) if isinstance(viewport, dict) else 16,
    })


@lru_cache(maxsize=1)
def building_overview_index():
    buildings = gdf_buildings_display.to_crs(epsg=3857)
    west, south, east, north = buildings.total_bounds
    scale = 1600 / max(east - west, north - south)
    width = max(1, round((east - west) * scale))
    height = max(1, round((north - south) * scale))
    return rasterize(
        ((geometry, index) for index, geometry in enumerate(buildings.geometry, 1)),
        out_shape=(height, width),
        transform=from_bounds(west, south, east, north, width, height),
        fill=0,
        all_touched=True,
        dtype="uint32",
    )


@lru_cache(maxsize=5)
def building_overview_png(measure, scenario):
    if measure == "cost":
        values = gdf_buildings_display["Rep_Cost"]
        maximum = 1_000_000
        palette = ["#5e4fa2", "#3288bd", "#66c2a5", "#e6f598", "#fee08b", "#f46d43", "#9e0142"]
    else:
        values = gdf_buildings_display[f"{scenario}_mean_flood"]
        maximum = 3
        palette = ["#cce5ff", "#99ccff", "#66b2ff", "#3399ff", "#007acc", "#004c99"]
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    color_indices = (np.nan_to_num(np.clip(values / maximum, 0, 0.999999)) * len(palette)).astype(int)
    palette_rgb = np.array([
        [int(color[offset:offset + 2], 16) for offset in (1, 3, 5)]
        for color in palette
    ], dtype=np.uint8)
    lookup = np.zeros((len(values) + 1, 4), dtype=np.uint8)
    lookup[1:, :3] = palette_rgb[color_indices]
    lookup[1:, 3] = 255
    lookup[1:][~np.isfinite(values), :3] = [140, 150, 159]
    rgba = lookup[building_overview_index()]
    height, width = rgba.shape[:2]
    scanlines = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )

###############################################
# Set working space
#working_dir = r"C:\Users\QuyenN\OneDrive - ESNZ\Offline_work\2_GNS\8_SpatialInvestmentTool\Codes"
#working_dir = r"C:\Users\Quyen\OneDrive - ESNZ\Offline_work\2_GNS\8_SpatialInvestmentTool\Codes"
#working_dir = r"C:\Users\qnguy\OneDrive - ESNZ\Offline_work\2_GNS\8_SpatialInvestmentTool\Codes"

#os.chdir(working_dir)

###############################################
# Data 
kelp_forest_services = {
    "Aesthetic_information": 7248,
    "Air_quality_regulation": 14064,
    "Climate_regulation": 301,
    "Erosion_prevention": 11730,
    "Existence_bequest_values": 451,
    "Food": 28320,
    "Information_for_cognitive_development": 3102,
    "Inspiration_for_culture_art_design": 0.32,
    "Maintenance_of_genetic_diversity": 58,
    "Maintenance_of_life_cycles": 4456,
    "Maintenance_of_soil_fertility": 5091,
    "Medicinal_resources": 1206,
    "Moderation_of_extreme_events": 810,
    "Recreation_and_tourism": 43008,
    "Raw_materials": 7047,
    "Waste_treatment": 341,
    "Water": 27950
}

saltmarsh_services = {
    "Aesthetic_information": 1252,
    "Air_quality_regulation": 351,
    "Climate_regulation": 241,
    "Existence_bequest_values": 1048,
    "Food": 3076,
    "Genetic_resources": 25,
    "Information_for_cognitive_development": 2139,
    "Inspiration_for_culture_art_design": 365,
    "Maintenance_of_genetic_diversity": 2822,
    "Maintenance_of_life_cycles": 2038,
    "Maintenance_of_soil_fertility": 10283,
    "Moderation_of_extreme_events": 13661,
    "Opportunities_for_recreation_and_tourism": 3292,
    "Raw_materials": 1024,
    "Regulation_of_water_flows": 63,
    "Spiritual_experience": 8,
    "Waste_treatment": 23045,
    "Water_Biomass_provisioning_services": 1869
}

exotic_trees_services = {
    "Aesthetic_information": 5100,
    "Air_quality_regulation": 9400,
    "Climate_regulation": 6200,
    "Erosion_prevention": 15800,
    "Existence_bequest_values": 2100,
    "Food": 8700,
    "Information_for_cognitive_development": 1400,
    "Inspiration_for_culture_art_design": 310,
    "Maintenance_of_genetic_diversity": 420,
    "Maintenance_of_life_cycles": 5100,
    "Maintenance_of_soil_fertility": 7600,
    "Moderation_of_extreme_events": 17600,
    "Recreation_and_tourism": 14800,
    "Raw_materials": 4300,
    "Waste_treatment": 12100,
    "Water": 18200
}

native_trees_services = {
    "Aesthetic_information": 8200,
    "Air_quality_regulation": 15200,
    "Climate_regulation": 9800,
    "Erosion_prevention": 21400,
    "Existence_bequest_values": 4100,
    "Food": 18750,
    "Information_for_cognitive_development": 2600,
    "Inspiration_for_culture_art_design": 540,
    "Maintenance_of_genetic_diversity": 980,
    "Maintenance_of_life_cycles": 7300,
    "Maintenance_of_soil_fertility": 11200,
    "Moderation_of_extreme_events": 28900,
    "Recreation_and_tourism": 22400,
    "Raw_materials": 6100,
    "Waste_treatment": 19800,
    "Water": 26500
}

mangrove_services = {
    "Aesthetic_information": 1200,
    "Air_quality_regulation": 3500,
    "Climate_regulation": 18500,
    "Erosion_prevention": 42000,
    "Existence_bequest_values": 2800,
    "Food": 14500,
    "Information_for_cognitive_development": 900,
    "Inspiration_for_culture_art_design": 250,
    "Maintenance_of_genetic_diversity": 600,
    "Maintenance_of_life_cycles": 9800,
    "Maintenance_of_soil_fertility": 11200,
    "Moderation_of_extreme_events": 65000,
    "Recreation_and_tourism": 16800,
    "Raw_materials": 4200,
    "Waste_treatment": 38000,
    "Water": 29000
}

ecosystem_services = {
    "Saltmarsh": saltmarsh_services,   
    "Mangrove": mangrove_services,    
    "Native_Trees": native_trees_services,
    "Exotic_Trees": exotic_trees_services,
    "Kelp_Forest": kelp_forest_services

}

default_nbs_reduction = {
    "Mangrove": {"Flood": 40, "River_Flood": 5, "Tsunami": 20, "Erosion": 45, "Landslide": 5},
    "Saltmarsh": {"Flood": 20, "River_Flood": 2, "Tsunami": 10, "Erosion": 35, "Landslide": 0},
    "Kelp_Forest": {"Flood": 10, "River_Flood": 1, "Tsunami": 5, "Erosion": 15, "Landslide": 0},
    "Native_Trees": {"Flood": 5, "River_Flood": 10, "Tsunami": 0, "Erosion": 10, "Landslide": 50},
    "Exotic_Forest": {"Flood": 5, "River_Flood": 8, "Tsunami": 0, "Erosion": 5, "Landslide": 40},
    "Mixed Native Shrub": {"Flood": 15, "River_Flood": 2, "Tsunami": 5, "Erosion": 20, "Landslide": 10}
}

# Life cycle costs (NZ$/ha) - based on research data from MfE 2023, Motu 2017, TTT 2025, KMR 2025
establishment_costs = {
    "Mangrove": 25000,  # Wetland Restoration mid-range ($2,000 - $50,000/ha)
    "Saltmarsh": 28000,  # Riparian Planting mid-range ($22,000 - $34,000/ha)
    "Kelp_Forest": 120000,  # Marine restoration (specialist)
    "Native_Trees": 11000,  # Forest & Native Vegetation Restoration mid-range ($2,500 - $20,000/ha)
    "Exotic_Forest": 10000,  # Forest & Native Vegetation Restoration lower range
    "Mixed Native Shrub": 11500  # Erosion Control mid-range ($10,000 - $13,000/ha)
}

# Annual maintenance costs (NZ$/ha/year)
maintenance_costs = {
    "Mangrove": 800,  # Wetland maintenance estimate
    "Saltmarsh": 1500,  # Riparian maintenance estimate
    "Kelp_Forest": 1500,  # Marine maintenance estimate
    "Native_Trees": 1200,  # Forest maintenance based on establishment recovery
    "Exotic_Forest": 1000,  # Plantation maintenance
    "Mixed Native Shrub": 1200  # Restoration maintenance
}

lcdb_colors = {
    "Estuarine Open Water": "#4575b4",
    "Herbaceous Saline Vegetation": "#c6dbef",
    "Exotic Forest": "#4dac26",
    "Gorse and/or Broom": "#d73027",
    "Gravel or Rock": "#f0e6d2",
    "Urban Parkland/Open Space": "#c2c2c2",
    "Orchard, Vineyard or Other Perennial Crop": "#e5c494",
    "Indigenous Forest": "#1b7837",
    "Built-up Area (settlement)": "#999999",
    "Forest - Harvested": "#7fbf7b",
    "High Producing Exotic Grassland": "#fee08b",
    "Deciduous Hardwoods": "#41ab5d",
    "Lake or Pond": "#74add1",
    "Short-rotation Cropland": "#d9ad7c",
    "Low Producing Grassland": "#fdae61",
    "River": "#abd9e9",
    "Transport Infrastructure": "#e31a1c",
    "Sand or Gravel": "#f7f7f7",
    "Manuka and/or Kanuka": "#99d594",
    "Mixed Exotic Shrubland": "#f46d43",
    "Broadleaved Indigenous Hardwoods": "#66c2a5",
    "Herbaceous Freshwater Vegetation": "#a6cee3"
}

# River flood map 
processed_raster_dir = Path(__file__).resolve().parent / "Data and Modelling" / "Processing_result"
river_flood_scenario_paths = {
    scenario: processed_raster_dir / f"{scenario}_napier_coarse_4326.tif"
    for scenario in ["0C", "1C", "2C", "3C"]
}

#Hazard map
hazard_map = {"Flood": "Coastal Flooding",
              "River_Flood": "River Flooding" ,
               "Tsunami": "Tsunami",
               "Erosion":"Coastal Erosion",
               "Landslides": "Landslide"
              }
###############################################
# Load LCDB
# --- Napier  ---
gdf_napier = gpd.read_file("Assets/Napier.gpkg").to_crs(epsg=4326)
gdf_napier["geometry"] = gdf_napier.geometry.simplify(0.0005)
gdf_napier["id"] = gdf_napier.index.astype(str)
gdf_napier["name"] = gdf_napier['TA2023_V_2'].astype(str)
napier_geojson = gdf_napier.__geo_interface__

# --- LCDB ---
gdf_lcdb = gpd.read_file("Assets/Napier_LCDB.gpkg").to_crs(epsg=4326)
gdf_lcdb["geometry"] = gdf_lcdb.geometry.simplify(0.001)
gdf_lcdb["id"] = gdf_lcdb.index.astype(str)
gdf_lcdb["Name_2018"] = gdf_lcdb["Name_2018"].astype(str)
#print(list(gdf_lcdb["Name_2018"].unique()))
lcdb_geojson = gdf_lcdb.__geo_interface__

# --- Buildings ---
gdf_buildings = gpd.read_file("Assets/River_flooding/building_exposed_river_flooding.gpkg")
gdf_buildings_projected = gdf_buildings.to_crs(epsg=2193).copy()
gdf_buildings_projected["geometry"] = gdf_buildings_projected.centroid
gdf_buildings_projected = gdf_buildings_projected[["Building_id", "geometry", "Rep_Cost",
                                                 "0C_mean_flood","1C_mean_flood","2C_mean_flood","3C_mean_flood"]]
gdf_buildings = gdf_buildings_projected.to_crs(epsg=4326)
center_lat = float(gdf_buildings.geometry.y.median())
center_lon = float(gdf_buildings.geometry.x.median())

building_footprints_path = (
    Path(__file__).resolve().parent
    / "Data and Modelling"
    / "Boundary"
    / "Hawkes_Bay_building.gpkg"
)
napier_bounds_2193 = gdf_napier.to_crs(epsg=2193).total_bounds
gdf_building_footprints = gpd.read_file(
    building_footprints_path,
    layer="buildings",
    bbox=tuple(napier_bounds_2193),
)
flood_attribute_columns = [
    "Building_id",
    "0C_mean_flood",
    "1C_mean_flood",
    "2C_mean_flood",
    "3C_mean_flood",
]
gdf_building_footprints = gdf_building_footprints.drop_duplicates(
    subset=["Building_id"]
)
gdf_buildings_display = gdf_building_footprints.merge(
    gdf_buildings[flood_attribute_columns].drop_duplicates(subset=["Building_id"]),
    on="Building_id",
    how="inner",
)
gdf_buildings_display = gdf_buildings_display[
    ["Building_id", "Rep_Cost", *flood_attribute_columns[1:], "geometry"]
]
gdf_buildings_display["geometry"] = gdf_buildings_display.geometry.simplify(
    1.0,
    preserve_topology=True,
)
gdf_buildings_display = gdf_buildings_display.to_crs(epsg=4326)
overview_west, overview_south, overview_east, overview_north = gdf_buildings_display.total_bounds
building_overview_bounds = [
    [float(overview_south), float(overview_west)],
    [float(overview_north), float(overview_east)],
]
napier_west, napier_south, napier_east, napier_north = gdf_napier.total_bounds
initial_map_bounds = [
    [float(napier_south), float(napier_west)],
    [float(napier_north), float(napier_east)],
]
initial_flood_raster, flood_raster_bounds = raster_overlay_data_uri("1C")

#gdf_buildings = gpd.read_file("Assets/Napier_building_4326.gpkg")
#gdf_buildings["geometry"] = gdf_buildings.centroid
#gdf_buildings = gdf_buildings[["Building_id", "geometry", "Rep_Cost",
#                               "0C_mean_flood","1C_mean_flood","2C_mean_flood","3C_mean_flood"]]

#---Flood damage -----
#building_flood_damages = pd.read_csv("Assets/River_flooding/building_exposed_river_flooding.csv")
#gdf_buildings= pd.merge(gdf_buildings,
#                        building_flood_damages[["Building_id", 
#                                                "0C_mean_flood","1C_mean_flood","2C_mean_flood","3C_mean_flood"]],
#                                                how ='left',
#                                               left_on = 'Building_id',
#                                                right_on = 'Building_id')
#gdf_buildings.to_file("Assets/River_flooding/building_exposed_river_flooding.gpkg")
###############################################
# Load Project Area Layers
project_path = "Assets/Project_Area"

project_layers = {}
options = ["Estuary", "LargestOffshore", "ModerateOffshore","PresentDay", "SmallOffshore"]

for name in options:
    layer = gpd.read_file(f"{project_path}/{name}.gpkg")
    layer = layer.to_crs(epsg=4326)
    layer["geometry"] = layer.geometry.simplify(0.0005)
    layer["id"] = layer.index.astype(str)
    project_layers[name] = {
        "name": f"{project_path}/{name}.gpkg",
        "gdf": layer,
        "geojson": json.loads(layer.to_json())
    }

###############################################
# LOAD CARBON SEQUESTRATION 
# Carbon sequestration table 
INTERVENTION_SPECIES_MAP = {
    "Mangrove": ["mangrove"],          # no data yet
    "Saltmarsh": ["saltmarsh"],         # no data yet
    "Kelp_Forest": ["kelp_forest"],       # no data yet
    "Mixed Native Shrub": ["mixed_shrubs"],
    "Native_Trees": [
        "totara", "kauri"
    ],
    "Exotic_Forest": [
        "other_conifers", "pine", "eucalyptus", "radiata_pine"
    ]
}
#Read data 
years = np.arange(1, 121)
# Growth curves (extended flat after max defined)
growth_curves = {
    "totara": [10.1,24.7,38.6,51.9,63.3,72.8,80.4,86.7,91.8,95.6,98.1,100],
    "kauri": [6.1,18.1,31.4,43.0,53.1,60.5,66.3,70,73,76,80,85,90,95,100],
    "kahikatea": [4.8,14.8,26.3,37.4,47.0,54.8,61.5,67,72,77,82,87,92,97,100],
    "rimu": [7.1,20.5,35.4,48.8,60.6,70,77,83,88,92,95,97,99,100],
    "beech": [53.6,23.6,40.7,56.6,69.7,80.5,88.2,93.6,97.3,99.1,100],
    "saltmarsh": [4.82,26.41,39.03,47.99,54.94,60.61,65.41,69.57,73.24,76.52,79.49,82.19,84.69,86.99,89.14,91.15,93.04,94.82,96.5,98.1,99.62,100],
    "mangrove": [4.82,26.41,39.03,47.99,54.94,60.61,65.41,69.57,73.24,76.52,79.49,82.19,84.69,86.99,89.14,91.15,93.04,94.82,96.5,98.1,99.62,100],
    "kelp_forest": [100],
    "mixed_shrubs": list(np.linspace(0.27,100,20)),
    "other_conifers": [12.5,28.3,41.2,52.1,61.5,69.2,75.8,81.2,85.6,89.1,91.8,94,96,97.5,98.5,99.2,99.7,100],
    "pine": [15.2,32.1,47.5,59.8,69.5,77.2,83.1,87.8,91.2,93.8,95.6,97,98.1,98.9,99.5,99.8,100],
    "eucalyptus": [18.5,35.2,49.3,61.1,70.5,77.8,83.5,88.2,91.9,94.6,96.5,97.8,98.7,99.3,99.8,100],
    "radiata_pine": [16.1,33.5,48.9,61.2,71.1,78.5,83.9,88.2,91.5,94,95.8,97.1,98.2,99,99.6,100]
}

# Final carbon values (plateau after maturity)
final_values = {
    "totara": 15.8,
    "kauri": 20.5,
    "kahikatea": 16.6,
    "rimu": 15.4,
    "beech": 22.0,
    "saltmarsh": 1.07,
    "mangrove": 1.53,
    "kelp_forest": 52,
    "mixed_shrubs": 17.8,
    "other_conifers": 18.5,
    "pine": 19.2,
    "eucalyptus": 17.8,
    "radiata_pine": 18.9
}

rows = []

for species, growth_list in growth_curves.items():
    growth = np.array(growth_list)

    # extend to 120 years
    if len(growth) < 120:
        growth = np.concatenate([growth, np.full(120-len(growth), growth[-1])])

    final = final_values[species]
    yearly = final * (growth / 100)

    rows.append([species, "growth %"] + list(growth[:120]))
    rows.append([species, "final carbon"] + [final]*120)
    rows.append([species, "yearly carbon"] + list(yearly[:120]))

columns = ["Type","Metric"] + [i for i in range(1,121)]
carbon_master = pd.DataFrame(rows, columns=columns)

#carbon_master.to_csv("Assets/carbon_sequestration_full_120yr.csv", index=False)
carbon_sequestration_data = carbon_master[carbon_master["Metric"]=="yearly carbon"]
carbon_sequestration_data.index = carbon_sequestration_data["Type"] 
carbon_sequestration_data = carbon_sequestration_data.drop(columns =["Metric","Type"])
carbon_sequestration_data =carbon_sequestration_data.to_dict()

growth_data = carbon_master[carbon_master["Metric"]=="growth %"]
growth_data.index = growth_data["Type"]
growth_data = growth_data.drop(columns =["Metric","Type"])
tree_growth_curves = growth_data.to_dict()
###############################################
# Functions 
def calculate_project_area(selected_project):
    project_data = project_layers[selected_project]
    project_gdf =  project_data['gdf']
    project_gdf_proj = project_gdf.to_crs(epsg=2193)
    area_m2 = project_gdf_proj.geometry.area.sum()
    area_ha = area_m2 / 10000
    return area_ha

def calculate_hazard_reduction(
    year,
    species,
    impact_area_ha,
    growth_data,
    max_reduction_map
):
    # Get growth %
    growth_pct = growth_data.get(year, {}).get(species, 0)

    # Convert to 0–1
    maturity = growth_pct / 100

    # Max hazard reduction for species
    max_reduction = max_reduction_map.get(species, 0)

    # Final hazard reduction (%)
    hazard_reduction_pct = maturity * max_reduction

    # Impacted area protected (ha)
    protected_area = hazard_reduction_pct * impact_area_ha

    return {
        "hazard_reduction_pct": hazard_reduction_pct,
        "protected_area_ha": protected_area
    }

def hazard_timeseries(species,  growth_data, max_reduction_map,hazard_type):
    results = {}

    for year in growth_data.keys():
        growth_pct = growth_data[year].get(species)

        if growth_pct is None:
            continue
        maturity = growth_pct / 100
        max_reduction = max_reduction_map.get(hazard_type)
        hazard_pct = maturity * max_reduction
        results[year] = {
            "year": year,
            "growth_pct" : growth_pct,
            "maturity" :  maturity,
            "max_reduction":max_reduction,
            "hazard_reduction_pct": hazard_pct,
            "hazard_type" : hazard_type,
            }
    #print(results)

    return results

###############################################
# Dash App
@lru_cache(maxsize=3)
def build_basemap(selected_basemap):
    if selected_basemap == "aerial":
        return dl.TileLayer(
            url="https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad/{z}/{x}/{y}.webp?api=c01m5e5daq4xkw4t33cs9xacfq2",
            attribution="Aerial imagery &copy; Toit&#363; Te Whenua LINZ",
            maxNativeZoom=19,
            maxZoom=20,
        )
    if selected_basemap == "gray":
        return dl.TileLayer(
            url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attribution="Tiles &copy; Esri, HERE, Garmin, FAO, NOAA, USGS, EPA, NPS",
            maxNativeZoom=16,
            maxZoom=20,
        )
    return dl.LayerGroup(
        attribution="&copy; Toit&#363; Te Whenua LINZ, CC BY 4.0",
        eventHandlers={
            "add": assign(
                "async function(event) {"
                "const group = event.target;"
                "if (group._linzLayer || group._linzLoading) return;"
                "group._linzLoading = true;"
                "const load = url => new Promise((resolve, reject) => {"
                "const script = document.createElement('script');"
                "script.src = url; script.onload = resolve; script.onerror = reject;"
                "document.head.appendChild(script);"
                "});"
                "try {"
                "if (!window.linzRendererReady) {"
                "window.linzRendererReady = (async () => {"
                "if (!window.L) await load('https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js');"
                "await load('https://cdn.jsdelivr.net/npm/maplibre-gl@5.6.2/dist/maplibre-gl.js');"
                "await load('https://cdn.jsdelivr.net/npm/@maplibre/maplibre-gl-leaflet@0.1.0/leaflet-maplibre-gl.js');"
                "})();"
                "}"
                "await window.linzRendererReady;"
                "if (!group._map) return;"
                "group._linzLayer = window.L.maplibreGL({"
                "style: 'https://basemaps.linz.govt.nz/v1/styles/topolite-v2.json?api=c01m5e5daq4xkw4t33cs9xacfq2&tileMatrix=WebMercatorQuad',"
                "interactive: false,"
                "pane: 'tilePane',"
                "canvasContextAttributes: {preserveDrawingBuffer: true}"
                "});"
                "group._linzLayer._transformGL = function(gl) {"
                "const center = this._map.getCenter();"
                "gl.jumpTo({center: [center.lng, center.lat], zoom: this._map.getZoom() - 1});"
                "};"
                "group._linzLayer._resize = function() {"
                "if (!this._map || !this._glMap) return;"
                "const size = this.getSize();"
                "this._container.style.width = size.x + 'px';"
                "this._container.style.height = size.y + 'px';"
                "this._glMap.resize();"
                "this._update();"
                "};"
                "group.addLayer(group._linzLayer);"
                "group._linzObserver = new ResizeObserver(() => {"
                "if (!group._map || !group._linzLayer._map) return;"
                "group._map.invalidateSize({pan: false});"
                "group._linzLayer._resize();"
                "});"
                "group._linzObserver.observe(group._map.getContainer());"
                "} catch (error) {"
                "window.linzRendererReady = null;"
                "console.error('LINZ basemap failed to load. Select the Esri fallback.', error);"
                "} finally { group._linzLoading = false; }"
                "}"
            ),
            "remove": assign(
                "function(event) { event.target._linzObserver?.disconnect(); }"
            ),
        },
    )


app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/maplibre-gl@5.6.2/dist/maplibre-gl.css",
    ],
)
app.layout = html.Div([
    html.H1(
                "Nature Based Solutions Spatial Investment Tool",
                style={"textAlign": "center", "marginBottom": "10px"}
            ),
    dcc.Tabs(id="tabs", value="tab-map", children=[

        # =========================
        # 🗺 MAP TAB
        # =========================
        dcc.Tab(label="🗺 Spatial Map", value="tab-map", children=[



            html.Div([

                # -------- LEFT: MAP --------
                html.Div([

                    html.Div([
                        html.Div([
                            html.Label("Layer", htmlFor="layer_selector", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="layer_selector",
                                options=[
                                    {"label": "Property / Buildings", "value": "buildings"},
                                    {"label": "Land Use (LCDB 2018)", "value": "lcdb"},
                                ],
                                value="buildings",
                                clearable=False,
                            ),
                        ]),
                        html.Div([
                            html.Label("Flood scenario", htmlFor="flood_scenario_selector", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="flood_scenario_selector",
                                options=[
                                    {"label": s, "value": s}
                                    for s in ["show replacement costs only", "0C", "1C", "2C", "3C"]
                                ],
                                value="1C",
                                clearable=False,
                            ),
                        ]),
                        html.Div([
                            html.Label("Basemap", htmlFor="basemap_selector", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="basemap_selector",
                                options=[
                                    {"label": "LINZ Topolite", "value": "topolite"},
                                    {"label": "LINZ aerial imagery", "value": "aerial"},
                                    {"label": "Esri light gray", "value": "gray"},
                                ],
                                value="topolite",
                                clearable=False,
                                searchable=False,
                            ),
                        ]),
                    ], className="map-selectors"),

                    html.Div(
                        id="map_summary",
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "8px 18px",
                            "padding": "8px 12px",
                            "marginTop": "8px",
                            "marginBottom": "6px",
                            "backgroundColor": "rgba(255,255,255,0.94)",
                            "border": "1px solid #b8c1c8",
                            "borderRadius": "4px",
                            "fontWeight": "600",
                        },
                    ),
                    dl.Map(
                        [
                            dl.LayerGroup(
                                id="basemap_layer",
                                children=build_basemap("topolite"),
                            ),
                            dl.LayersControl(
                                [
                                    dl.Overlay(
                                        dl.TileLayer(
                                            url="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
                                            attribution="Hillshade &copy; Esri",
                                            pane="tilePane",
                                            zIndex=220,
                                            opacity=0.3,
                                            maxNativeZoom=16,
                                            maxZoom=20,
                                        ),
                                        name=cast(Any, "Terrain relief (Esri)"),
                                        checked=False,
                                    ),
                                    dl.Overlay(
                                        dl.ImageOverlay(
                                            id="flood_raster",
                                            url=initial_flood_raster,
                                            bounds=flood_raster_bounds,
                                            opacity=0.68,
                                            zIndex=300,
                                        ),
                                        name=cast(Any, "River flood depth"),
                                        checked=True,
                                    ),
                                    dl.Overlay(
                                        dl.LayerGroup([
                                            dl.ImageOverlay(
                                                id="building_overview",
                                                url="/api/building-overview/depth/1C.png",
                                                bounds=building_overview_bounds,
                                                opacity=1,
                                                zIndex=310,
                                                interactive=False,
                                            ),
                                            dl.GeoJSON(
                                            id="buildings_geojson",
                                            eventHandlers={
                                                "layeradd": assign(
                                                    "function(event) {"
                                                    "const renderer = event.layer?._renderer;"
                                                    "if (!renderer?._ctx || renderer._buildingShadowHook) return;"
                                                    "renderer._buildingShadowHook = true;"
                                                    "renderer._container.classList.add('building-shadow-canvas');"
                                                    "const drawPolygon = renderer._updatePoly;"
                                                    "renderer._updatePoly = function(layer, closed) {"
                                                    "if (!layer.feature?.properties?.Building_id || this._map.getZoom() < 17) {"
                                                    "return drawPolygon.call(this, layer, closed);"
                                                    "}"
                                                    "const context = this._ctx;"
                                                    "context.save();"
                                                    "context.shadowColor = 'rgba(20, 30, 38, 0.38)';"
                                                    "context.shadowBlur = 3;"
                                                    "context.shadowOffsetX = 3;"
                                                    "context.shadowOffsetY = 4;"
                                                    "try { return drawPolygon.call(this, layer, closed); }"
                                                    "finally { context.restore(); }"
                                                    "};"
                                                    "renderer._redraw();"
                                                    "}"
                                                ),
                                                "add": assign(
                                                    "function(event) {"
                                                    "const group = event.target;"
                                                    "const layer = group.getLayers().find(item => item._renderer?._ctx);"
                                                    "if (layer) group.fire('layeradd', {layer: layer});"
                                                    "}"
                                                ),
                                            },
                                            hoverStyle={
                                                "color": "#172f3c",
                                                "weight": 2,
                                                "fillOpacity": 1,
                                            },
                                            style=assign(
                                                "function(feature) {"
                                                "const p = feature.properties;"
                                                "return {color: '#39434d', weight: 0.45, "
                                                "fillColor: p.fill_color || '#8c969f', fillOpacity: 0.82};"
                                                "}"
                                            ),
                                            onEachFeature=assign(
                                                "function(feature, layer) {"
                                                "const p = feature.properties;"
                                                "layer.bindTooltip('Building: ' + p.Building_id "
                                                "+ '<br>' + p.display_text, {className: 'building-tooltip', sticky: true});"
                                                "}"
                                            ),
                                            ),
                                        ]),
                                        name=cast(Any, "Building footprints"),
                                        id="building_overlay",
                                        checked=True,
                                    ),
                                    dl.Overlay(
                                        dl.GeoJSON(
                                            id="lcdb_geojson",
                                            data=EMPTY_GEOJSON,
                                            hideout={"colors": lcdb_colors},
                                            style=assign(
                                                "function(feature, context) {"
                                                "const color = context.hideout.colors[feature.properties.Name_2018] || '#b8c1c8';"
                                                "return {color: '#ffffff', weight: 0.25, fillColor: color, fillOpacity: 0.82};"
                                                "}"
                                            ),
                                        ),
                                        name=cast(Any, "Land cover"),
                                        id="land_cover_overlay",
                                        checked=False,
                                    ),
                                    dl.Overlay(
                                        dl.GeoJSON(
                                            id="project_geojson",
                                            data=EMPTY_GEOJSON,
                                            options={"style": {
                                                "color": "#176b46",
                                                "weight": 3,
                                                "fillColor": "#28a745",
                                                "fillOpacity": 0.18,
                                            }},
                                        ),
                                        name=cast(Any, "Preset scenario"),
                                        checked=True,
                                    ),
                                    dl.Overlay(
                                        dl.GeoJSON(
                                            id="buffer_geojson",
                                            data=EMPTY_GEOJSON,
                                            options={"style": {
                                                "color": "#1672c4",
                                                "weight": 3,
                                                "fillColor": "#1e90ff",
                                                "fillOpacity": 0.25,
                                            }},
                                        ),
                                        name=cast(Any, "Project buffer"),
                                        checked=True,
                                    ),
                                    dl.Overlay(
                                        dl.GeoJSON(
                                            id="napier_geojson",
                                            data=napier_geojson,
                                            options={"style": {
                                                "color": "#c62828",
                                                "weight": 2,
                                                "fillOpacity": 0,
                                            }},
                                        ),
                                        name=cast(Any, "Napier boundary"),
                                        checked=True,
                                    ),
                                    dl.Overlay(
                                        dl.TileLayer(
                                            url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                                            attribution="Labels &copy; Esri",
                                            pane="overlayPane",
                                            zIndex=650,
                                            opacity=0.9,
                                            maxNativeZoom=16,
                                            maxZoom=20,
                                        ),
                                        name=cast(Any, "Place labels (Esri)"),
                                        checked=False,
                                    ),
                                ],
                                position="topright",
                                overlays=[
                                    "River flood depth",
                                    "Building footprints",
                                    "Land cover",
                                    "Preset scenario",
                                    "Project buffer",
                                    "Napier boundary",
                                ],
                            ),
                            dl.Colorbar(
                                id="map_legend",
                                position="bottomleft",
                                colorscale=["#cce5ff", "#99ccff", "#66b2ff", "#3399ff", "#007acc", "#004c99"],
                                min=0,
                                max=3,
                                width=210,
                                height=14,
                                tickValues=[0, 1, 2, 3],
                                unit=" m",
                                className="flood-depth-legend",
                                tooltip=True,
                            ),
                            dl.FeatureGroup([
                                EditControl(
                                    id="aoi_draw_control",
                                    draw={
                                        "polyline": False,
                                        "polygon": True,
                                        "rectangle": True,
                                        "circle": False,
                                        "marker": False,
                                        "circlemarker": False,
                                    },
                                    edit={"edit": True, "remove": True},
                                )
                            ]),
                        ],
                        id="leaflet_map",
                        center=cast(Any, {
                            "lat": float((napier_south + napier_north) / 2),
                            "lng": float((napier_west + napier_east) / 2),
                        }),
                        bounds=cast(Any, initial_map_bounds),
                        boundsOptions={"padding": cast(Any, [16, 16])},
                        zoomSnap=0.5,
                        trackViewport=True,
                        preferCanvas=True,
                        className="spatial-leaflet-map",
                        style={"width": "100%"},
                    ),

                ], className="spatial-map-panel", style={"width": "70%", "display": "inline-block"}),

                # -------- RIGHT: CONTROL PANEL --------
                html.Div([

                    html.H3("Scenarios & Assumptions", style={"textAlign": "left"}),

                    # ---- Project ----
                    html.Label("Preset scenario:", htmlFor="project_selector", style={"fontWeight": "bold"}),

                    html.Div([
                        dcc.Dropdown(
                            id="project_selector",
                            options=[
                                {"label": name, "value": name} for name in options
                            ] + [{"label": "Custom AOI", "value": "Custom_AOI"}],
                            value="Estuary",
                            clearable=False,
                            style={"width": "100%", "minWidth": 0},
                        ),
                        html.Button(
                            html.Img(
                                src="https://cdn.jsdelivr.net/npm/lucide-static@1.66.0/icons/vector-square.svg",
                                alt="",
                                width=20,
                                height=20,
                            ),
                            id="draw_custom_aoi",
                            n_clicks=0,
                            type="button",
                            title="Draw custom AOI",
                            className="digitize-aoi-button",
                            **cast(dict[str, Any], {"aria-label": "Draw custom AOI"}),
                        ),
                    ], className="preset-scenario-controls"),

                    dcc.Store(id="custom_aoi_geojson"),

                    html.Br(),

                    html.Label("Buffer Distance (m):", style={"fontWeight": "bold"}),

                    dcc.Slider(
                        id="buffer_slider",
                        min=10,
                        max=2000,
                        step=10,
                        value=500,
                        marks={
                            10: "10 m",
                            500: "500 m",
                            1000: "1 km",
                            1500: "1.5 km",
                            2000: "2 km",
                        },
                    ),

                    html.Hr(),

                    # ---- Hidden fields ----
                    dcc.Input(id="project_impact_area_ha", type="number", style={"display": "none"}),
                    dcc.Input(id="project_area_ha", type="number", style={"display": "none"}),
                    dcc.Input(id="n_houses", type="number", style={"display": "none"}),
                    dcc.Input(id="building_values", type="number", style={"display": "none"}),
                    dcc.Input(id="total_damage", type="number", style={"display": "none"}),
                    # ---- NbS ----
                    html.Label("Nature-Based Intervention:", style={"fontWeight": "bold"}),

                    dcc.Dropdown(
                        id="intervention_type",
                        options=[
                            {"label": "Mangrove", "value": "Mangrove"},
                            {"label": "Saltmarsh", "value": "Saltmarsh"},
                            {"label": "Kelp Forest", "value": "Kelp_Forest"},
                            {"label": "Native Trees", "value": "Native_Trees"},
                            {"label": "Exotic Trees", "value": "Exotic_Forest"},
                            {"label": "Mixed Native Shrubs", "value": "Mixed Native Shrub"}
                        ],
                        value="Saltmarsh",
                        clearable=False
                    ),

                    html.Br(),

                    html.Label("Please choose species:", style={"fontWeight": "bold"}),

                    dcc.Dropdown(
                            id="carbon_species",
                            options=[],
                            value=None,
                            clearable=False,
                            style={"width": "50%", "marginBottom": "15px"}
                        ),
                    html.Br(),
                    # ---- Hazards ----
                    html.Label("Hazards:", style={"fontWeight": "bold"}),

                    dcc.Dropdown(
                        id="hazard_type",
                        options=[
                            {"label": "Coastal Flooding", "value": "Flood"},
                            {"label": "River Flooding", "value": "River_Flood"},
                            {"label": "Tsunami", "value": "Tsunami"},
                            {"label": "Coastal Erosion", "value": "Erosion"},
                            {"label": "Landslides", "value": "Landslide"}
                        ],
                        value=["River_Flood"],
                        multi=True
                    ),

                    html.Div(id="hazard_reduction_container"),

                    html.Hr(),

                    # ---- Economic inputs ----
                    html.H4("Economic Assumptions", style={
                        "marginBottom": "15px",
                        "borderBottom": "2px solid #ddd",
                        "paddingBottom": "5px"
                    }),

                    # ---- GRID LAYOUT ----
                    html.Div([

                        # ---- Timeline ----
                        html.Label("Project Timeline (years)", style={"alignSelf": "center"}),
                        dcc.Input(
                            id="project_year",
                            type="number",
                            value=50,
                            style={"width": "100%"}
                        ),

                        # ---- Carbon Price ----
                        html.Label("Carbon Price ($/tCO₂)", style={"alignSelf": "center"}),
                        dcc.Input(
                            id="carbon_price",
                            type="number",
                            value=43.82,
                            style={"width": "100%"}
                        ),

                        # ---- Discount Rate ----
                        html.Label("Discount Rate (%)", style={"alignSelf": "center"}),
                        dcc.Input(
                            id="discount_rate",
                            type="number",
                            value=0.03,
                            step=0.005,
                            style={"width": "100%"}
                        ),

                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",  # 🔥 label | input
                        "rowGap": "12px",
                        "columnGap": "10px",
                        "alignItems": "center"
                    }),

                    html.Div(
                        id="results_output",
                        style={"fontWeight": "bold", "whiteSpace": "pre-line"}
                    ),

                ], className="spatial-assumptions-panel", style={
                    "width": "28%",
                    "display": "inline-block",
                    "padding": "20px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "10px"
                })

            ],
            className="spatial-layout",
            style={
                "display": "flex",          
                "flexDirection": "row",     # force side-by-side
                "alignItems": "flex-start"  # align top
            })
        ]),

        # =========================
        # 🌱 CARBON TAB
        # =========================
        dcc.Tab(label="🌱 Carbon Sequestration", value="tab-carbon", children=[

            html.Div([
                html.H3("Current Annual Carbon Sequestration (tCO₂/ha/year)"),
                dcc.Graph(id="carbon_chart"),
            ]),

            html.Div([
                html.H3("Yearly Carbon Sequestration Per Ha"),
                html.Div(id="carbon_npv_output", style={"marginTop": "15px", 
                                                    "fontWeight": "bold", "fontSize": "20px"}),
                html.Div(id="carbon_table")
            ], style={"padding": "18px"}),

        ]),
        # =========================
        # 🌊 HAZARD TAB
        # =========================
        dcc.Tab(label="🌊 Hazard Reduction", value="tab-hazard", children=[


            html.Div([
                html.H3("Hazard Reduction for Project Impact Area (%)"),
                dcc.Graph(id="hazard_reduction_chart"),
            ]),

             html.Div([
                html.H3("Yearly Hazard Reduction for Project Impact Area (property value NZ$ protected)"),
                html.Div(id="hazard_npv_summary"),
                html.Div(id="hazard_reduction_table"),
            ], style={"padding": "18px"}),           
        ]),
        # =========================
        # 💰 COST TAB
        # =========================
        dcc.Tab(label="💰 Life-Cycle Costs", value="tab-cost", children=[
            html.Div([
                # Left: Cost Inputs
                html.Div([
                    html.H3("Cost Inputs", style={"textAlign": "center"}),
                    html.Label("Establishment cost (NZD/ha):", style={"fontWeight": "bold", "marginTop": "10px"}),
                    dcc.Input(id="establishment_cost_input", type="number", value=100000, style={"width": "100%", "marginBottom": "20px"}),
                    html.Label("Annual maintenance cost (NZD/ha/yr):", style={"fontWeight": "bold"}),
                    dcc.Input(id="maintenance_cost_input", type="number", value=1000, style={"width": "100%", "marginBottom": "20px"}),
                ], style={"width": "28%", "display": "inline-block", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px", "maxHeight": "80vh", "overflowY": "auto"}),
                # Right: Chart and Summary
                html.Div([
                    html.H3("Project Life Cycle Cost Analysis"),
                    dcc.Graph(id="cost_chart"),
                    html.Div(id="cost_summary", style={"padding": "18px", "backgroundColor": "#f5f5f5", "borderRadius": "8px", "marginTop": "20px"})
                ], style={"width": "68%", "display": "inline-block", "padding": "20px"})
            ], style={"display": "flex", "flexDirection": "row", "alignItems": "flex-start"})
        ]),

        # =========================
        # 📊 NPV TAB
        # =========================
        dcc.Tab(label="📊 NPV Analysis", value="tab-npv", children=[
            html.Div([
                html.Div([
                    # Left Panel - Project Summary
                    html.Div([
                        html.H3("📋 PROJECT SUMMARY RESULTS"),
                        html.Div(id="project_summary", style={
                            "backgroundColor": "#f5f5f5",
                            "borderRadius": "8px",
                            "padding": "18px",
                            "border": "2px solid #ddd"
                        })
                    ], style={
                        "flex": "1",
                        "marginRight": "18px",
                        "minWidth": "300px"
                    }),
                    # Right Panel - Waterfall Chart
                    html.Div([
                        html.H3("📊 NPV Waterfall Analysis"),
                        dcc.Graph(id="npv_waterfall",  
                        style={"height": "600px"}  # 👈 enforce at UI level
)
                    ], style={
                        "flex": "1.5",
                        "minWidth": "400px"
                    })
                ], style={
                    "display": "flex",
                    "gap": "18px",
                    "flexWrap": "wrap"
                })
            ], style={"padding": "18px"})
        ]),



    ])
])

###############################################
app.clientside_callback(
    """
    function(nClicks) {
        if (!nClicks) return window.dash_clientside.no_update;
        const map = document.getElementById('leaflet_map');
        if (map) map.scrollIntoView({behavior: 'smooth', block: 'center'});
        return {mode: 'polygon', n_clicks: nClicks};
    }
    """,
    Output("aoi_draw_control", "drawToolbar"),
    Input("draw_custom_aoi", "n_clicks"),
    prevent_initial_call=True,
)


@app.callback(
    Output("custom_aoi_geojson", "data"),
    Input("aoi_draw_control", "geojson"),
)
def store_custom_aoi(drawn_aoi):
    return drawn_aoi


@app.callback(
    Output("project_selector", "value"),
    Input("custom_aoi_geojson", "data"),
    prevent_initial_call=True,
)
def switch_to_custom_aoi_if_drawn(custom_aoi_geojson):
    if custom_aoi_geojson and custom_aoi_geojson.get("features"):
        return "Custom_AOI"
    return dash.no_update


@app.callback(
    Output("project_summary", "children"),
    Input("intervention_type", "value"),
    Input("hazard_type", "value"),
    Input("project_area_ha", "value"),
    Input("project_impact_area_ha", "value"),
    Input("n_houses", "value"),
    Input("total_damage", "value"),
    Input("carbon_species", "value"),
    Input("carbon_price", "value"),
    Input("project_year", "value"),
    Input("discount_rate", "value"),
    Input("establishment_cost_input", "value"),
    Input("maintenance_cost_input", "value")
)
def update_project_summary(
    intervention,
    selected_hazards,
    project_area,
    project_impact_area,
    num_properties,
    total_damage,
    carbon_species,
    carbon_price,
    project_year,
    discount_rate,
    establishment_cost_per_ha,
    maintenance_cost_per_ha,
):
    if not intervention or not selected_hazards:
        return html.Div("Select intervention type and hazards to view project summary")
    if project_area is None or project_area <= 0:
        return html.Div("Select a project or draw an AOI to view project summary.")
    
    if carbon_price is None:
        carbon_price = 0
    if discount_rate is None:
        discount_rate = 0.05
    if total_damage is None:
        total_damage = 0
    if project_year is None:
        project_year = 120
    if establishment_cost_per_ha is None:
        establishment_cost_per_ha = 100000
    if maintenance_cost_per_ha is None:
        maintenance_cost_per_ha = 1000
    
    # Format hazards as comma-separated list
    hazards_text = ", ".join(selected_hazards) if isinstance(selected_hazards, list) else selected_hazards
    
    # Format total damage as currency
    total_damage_str = f"NZ$ {total_damage/1e6:.2f} million" if total_damage else "N/A"

    total_establishment_cost = establishment_cost_per_ha * project_area
    years = np.arange(1, project_year + 1)
    annual_costs = np.ones(project_year) * maintenance_cost_per_ha * project_area
    discount_factors = 1 / (1 + discount_rate) ** years
    discounted_costs = annual_costs * discount_factors
    cumulative_maintenance_cost = np.sum(discounted_costs)
    total_costs = total_establishment_cost + cumulative_maintenance_cost
    
    # Calculate carbon benefits
    carbon_benefit = 0
    if carbon_species:
        sel_ages_orig, sel_values_orig = [], []
        for age in sorted(carbon_sequestration_data.keys()):
            val = carbon_sequestration_data[age].get(carbon_species, None)
            if val is not None:
                sel_ages_orig.append(age)
                sel_values_orig.append(val)
        
        if sel_ages_orig and project_area:
            f = interp1d(sel_ages_orig, sel_values_orig, kind='linear', fill_value='extrapolate')
            interp_ages = np.arange(1, project_year + 1)
            interp_values = np.maximum(f(interp_ages), 0)
            annual_value = interp_values * carbon_price
            annual_value_total = annual_value * project_area
            discount_factors_carbon = 1 / (1 + discount_rate) ** interp_ages
            npv_values = annual_value_total * discount_factors_carbon
            carbon_benefit = np.sum(npv_values)
    
    # Calculate hazard reduction benefits
    hazard_benefit = 0
    if selected_hazards and total_damage > 0:
        max_reduction_map = default_nbs_reduction.get(intervention, {})
        all_rows = []
        for hazard_type in selected_hazards:
            timeseries = hazard_timeseries(carbon_species, tree_growth_curves, max_reduction_map, hazard_type)
            for year, row in timeseries.items():
                all_rows.append(row)
        
        if all_rows:
            all_hazard_timeseries = pd.DataFrame(all_rows)
            table_hazard_timeseries = all_hazard_timeseries[all_hazard_timeseries["year"] <= project_year]
            
            for hazard_type in selected_hazards:
                filtered_df = table_hazard_timeseries[table_hazard_timeseries["hazard_type"] == hazard_type]
                for idx, row in filtered_df.iterrows():
                    year = row["year"]
                    hazard_reduction_pct = row["hazard_reduction_pct"]
                    protected_value = total_damage * (hazard_reduction_pct / 100) * 0.01
                    discount_factor = 1 / (1 + discount_rate) ** year
                    discounted_value = protected_value * discount_factor
                    hazard_benefit += discounted_value
    
    # Calculate total NPV and total benefit
    total_benefit = carbon_benefit + hazard_benefit
    total_npv = total_benefit - total_costs
    
    summary_items = [
        ("🌿 Project", "Estuary"),
        ("🌱 Ecosystem", intervention.replace("_", " ")),
        ("📌 Protected Hazards", hazards_text),
        ("📏 Project Area", f"{project_area:.2f} ha" if project_area else "N/A"),
        ("📏 Project Impact Area (buffered)", f"{project_impact_area:.2f} ha" if project_impact_area else "N/A"),
        ("� Project Timeline", f"{int(project_year)} years"),
        ("📈 Discount Rate", f"{discount_rate*100:.1f}%"),
        ("🏘️ Impacted Properties", f"{int(num_properties)} houses" if num_properties else "N/A"),
        ("💰 Total Replacement Cost within the impacted area", total_damage_str),
        ("", ""),  # Spacer
        ("🏗️ Total Project Cost", f"NZ$ {total_costs:,.0f}"),
        ("🌱 Total Benefit", f"NZ$ {total_benefit:,.0f}"),
        ("💵 Net Present Value (NPV)", f"NZ$ {total_npv:,.0f}"),
    ]
    
    return html.Div([
        html.Div([
            html.Div(label, style={
                "fontWeight": "bold",
                "fontSize": "14px",
                "marginTop": "12px",
                "color": "#333"
            }),
            html.Div(value, style={
                "fontSize": "16px",
                "color": "#0066cc" if value else "transparent",
                "marginBottom": "8px",
                "paddingLeft": "10px",
                "borderLeft": "3px solid #4CAF50" if value else "none"
            })
        ])
        for label, value in summary_items
    ])

@app.callback(
    Output("carbon_species", "options"),
    Output("carbon_species", "value"),
    Input("intervention_type", "value")
)
def update_species_dropdown(intervention_type):
    if intervention_type is None:
        return [], None
    
    # Get species for selected intervention
    species_list = INTERVENTION_SPECIES_MAP.get(intervention_type, [])
    
    # Create options for dropdown
    options = [
        {"label": s.replace("_", " ").title(), "value": s}
        for s in species_list
    ]
    
    # Set default value to first species in list
    default_value = species_list[0] if species_list else None
    
    return options, default_value

@app.callback(
    Output("hazard_reduction_container", "children"),
    Input("intervention_type", "value"),
    Input("hazard_type", "value")
)
def update_hazard_inputs(intervention, selected_hazards):

    components = []

    if not selected_hazards:
        return components

    # Human-readable label
    label_map = {
                "Flood": "Coastal Flooding",
                "River_Flood": "River Flooding",
                "Tsunami": "Tsunami",
                "Erosion": "Coastal Erosion",
                "Landslide": "Landslides"
            }

    if "Flood" in selected_hazards:
        default_value = default_nbs_reduction.get(intervention, {}).get("Flood", 0)

        components.append(
            html.Div([
                html.Label("Coastal Flood Damage Reduction versus baseline (%)"),
                dcc.Input(
                    id="flood_reduction",
                    type="number",
                    min=0,
                    max=100,
                    step=5,
                    value=default_value,
                    style={"width": "100%", "marginBottom": "10px"}
                )
            ])
        )

    if "River_Flood" in selected_hazards:
        default_value = default_nbs_reduction.get(intervention, {}).get("River_Flood", 0)
        components.append(
            html.Div([
                html.Label("River Flood Damage Reduction versus baseline (%)"),
                dcc.Input(
                    id="river_flood_reduction",
                    type="number",
                    min=0,
                    max=100,
                    step=5,
                    value=default_value,
                    style={"width": "100%", "marginBottom": "10px"}
                )
            ])
        )

    if "Tsunami" in selected_hazards:
        default_value = default_nbs_reduction.get(intervention, {}).get("Tsunami", 0)

        components.append(
            html.Div([
                html.Label("Tsunami Damage Reduction versus baseline (%)"),
                dcc.Input(
                    id="tsunami_flood_reduction",
                    type="number",
                    min=0,
                    max=100,
                    step=5,
                    value=default_value,
                    style={"width": "100%", "marginBottom": "10px"}
                )
            ])
        )

    if "Erosion" in selected_hazards:
        default_value = default_nbs_reduction.get(intervention, {}).get("Erosion", 0)

        components.append(
            html.Div([
                html.Label("Coastal Erosion Damage Reduction versus baseline (%)"),
                dcc.Input(
                    id="erosion_reduction",
                    type="number",
                    min=0,
                    max=100,
                    step=5,
                    value=default_value,
                    style={"width": "100%", "marginBottom": "10px"}
                )
            ])
        )

    if "Landslide" in selected_hazards:
        default_value = default_nbs_reduction.get(intervention, {}).get("Landslide", 0)

        components.append(
            html.Div([
                html.Label("Landslide Damage Reduction versus baseline (%)"),
                dcc.Input(
                    id="landslide_reduction",
                    type="number",
                    min=0,
                    max=100,
                    step=5,
                    value=default_value,
                    style={"width": "100%", "marginBottom": "10px"}
                )
            ])
        )
    return components

def _building_geojson_json(west, south, east, north, scenario, measure, visible, zoom):
    if not visible or zoom < 15.5:
        return json.dumps(EMPTY_GEOJSON)

    buildings = gdf_buildings_display.cx[west:east, south:north].copy()
    if buildings.empty:
        return json.dumps(EMPTY_GEOJSON)

    if measure == "cost":
        value_column = "Rep_Cost"
        maximum = 1_000_000
        palette = ["#5e4fa2", "#3288bd", "#66c2a5", "#e6f598", "#fee08b", "#f46d43", "#9e0142"]
    else:
        value_column = f"{scenario}_mean_flood"
        maximum = 3
        palette = ["#cce5ff", "#99ccff", "#66b2ff", "#3399ff", "#007acc", "#004c99"]

    values = pd.to_numeric(buildings[value_column], errors="coerce")
    colors = []
    for value in values:
        if pd.isna(value):
            colors.append("#8c969f")
        else:
            index = min(int(np.clip(value / maximum, 0, 0.999999) * len(palette)), len(palette) - 1)
            colors.append(palette[index])
    buildings["fill_color"] = colors
    if measure == "cost":
        buildings["display_text"] = values.map(
            lambda value: f"Replacement value: NZ$ {value:,.0f}" if pd.notna(value) else "No replacement value"
        )
    else:
        buildings["display_text"] = values.map(
            lambda value: f"Mean flood depth: {value:.2f} m" if pd.notna(value) else "No flood-depth data"
        )
    return buildings.to_json(drop_id=True)


@lru_cache(maxsize=32)
def _cached_building_geojson(west, south, east, north, scenario, measure, visible, zoom):
    return _building_geojson_json(west, south, east, north, scenario, measure, visible, zoom)


@app.server.get("/api/building-overview/<measure>/<scenario>.png")
def building_overview_api(measure, scenario):
    if measure not in {"cost", "depth"} or scenario not in {"0C", "1C", "2C", "3C"}:
        return Response("Invalid map query", status=400, mimetype="text/plain")
    return Response(building_overview_png(measure, scenario), mimetype="image/png")


@app.callback(
    Output("building_overview", "url"),
    Output("building_overview", "opacity"),
    Input("flood_scenario_selector", "value"),
    Input("layer_selector", "value"),
    Input("leaflet_map", "zoom"),
)
def update_building_overview(scenario, selected_layer, zoom):
    measure = "cost" if scenario == "show replacement costs only" else "depth"
    raster_scenario = scenario if scenario in {"0C", "1C", "2C", "3C"} else "1C"
    opacity = 1 if selected_layer == "buildings" and (zoom is None or zoom < 15.5) else 0
    return f"/api/building-overview/{measure}/{raster_scenario}.png", opacity


@app.server.get("/api/building-footprints")
def building_footprints_api():
    try:
        bounds = [float(request.args[name]) for name in ("west", "south", "east", "north")]
        west, south, east, north = bounds
        scenario = request.args.get("scenario", "1C")
        measure = request.args.get("measure", "depth")
        visible = request.args.get("visible", "true").lower() == "true"
        zoom = float(request.args.get("zoom", "16"))
        if scenario not in {"0C", "1C", "2C", "3C"} or measure not in {"cost", "depth"}:
            raise ValueError("Invalid map query")
    except (KeyError, TypeError, ValueError):
        return Response("Invalid map query", status=400, mimetype="text/plain")

    geojson = _cached_building_geojson(
        round(west, 6), round(south, 6), round(east, 6), round(north, 6),
        scenario, measure, visible, zoom,
    )
    return Response(geojson, mimetype="application/geo+json")


@app.callback(
    Output("basemap_layer", "children"),
    Input("basemap_selector", "value"),
    prevent_initial_call=True,
)
def update_basemap(selected_basemap):
    return build_basemap(selected_basemap)


@app.callback(
    Output("layer_selector", "value"),
    Output("leaflet_map", "viewport"),
    Input("flood_scenario_selector", "value"),
    State("leaflet_map", "center"),
    State("leaflet_map", "zoom"),
    prevent_initial_call=True,
)
def focus_replacement_costs(scenario, map_center, map_zoom):
    if scenario != "show replacement costs only":
        return dash.no_update, dash.no_update
    if map_zoom is not None and map_zoom >= 15.5:
        return "buildings", dash.no_update
    if isinstance(map_center, dict):
        map_center = [map_center["lat"], map_center["lng"]]
    return "buildings", {
        "center": map_center if map_center is not None else [center_lat, center_lon],
        "zoom": 16,
    }


@app.callback(
    Output("land_cover_overlay", "checked"),
    Output("building_overlay", "checked"),
    Input("layer_selector", "value"),
)
def update_display_layer_visibility(selected_layer):
    return selected_layer == "lcdb", selected_layer == "buildings"


@app.callback(
    Output("map_legend", "colorscale"),
    Output("map_legend", "max"),
    Output("map_legend", "tickValues"),
    Output("map_legend", "tickText"),
    Output("map_legend", "unit"),
    Output("map_legend", "className"),
    Input("flood_scenario_selector", "value"),
)
def update_map_legend(scenario):
    if scenario == "show replacement costs only":
        return (
            ["#5e4fa2", "#3288bd", "#66c2a5", "#e6f598", "#fee08b", "#f46d43", "#9e0142"],
            1_000_000,
            [0, 500_000, 1_000_000],
            ["0", "500k", "1m+"],
            "",
            "replacement-value-legend",
        )
    return (
        ["#cce5ff", "#99ccff", "#66b2ff", "#3399ff", "#007acc", "#004c99"],
        3,
        [0, 1, 2, 3],
        ["0", "1", "2", "3+"],
        "",
        "flood-depth-legend",
    )


@app.callback(
    Output("buildings_geojson", "url"),
    Output("lcdb_geojson", "data"),
    Output("project_geojson", "data"),
    Output("buffer_geojson", "data"),
    Output("flood_raster", "url"),
    Output("flood_raster", "opacity"),
    Output("map_summary", "children"),
    Output("project_area_ha", "value"),
    Output("project_impact_area_ha", "value"),
    Output("n_houses", "value"),
    Output("building_values", "value"),
    Output("total_damage", "value"),
    Input("project_selector", "value"),
    Input("layer_selector", "value"),
    Input("flood_scenario_selector", "value"),
    Input("buffer_slider", "value"),
    Input("custom_aoi_geojson", "data"),
    Input("leaflet_map", "center"),
    Input("leaflet_map", "zoom"),
)
def update_map(
    selected_project,
    selected_layer,
    flood_scenario_selector,
    buffer_km,
    custom_aoi_geojson,
    map_center,
    map_zoom,
):
    viewport = {"center": map_center, "zoom": map_zoom}
    buildings_url = building_geojson_url(
        viewport,
        flood_scenario_selector,
        selected_layer,
    )
    lcdb_data = lcdb_geojson if selected_layer == "lcdb" else EMPTY_GEOJSON
    raster_scenario = (
        flood_scenario_selector
        if flood_scenario_selector in {"0C", "1C", "2C", "3C"}
        else "1C"
    )
    raster_url, _ = raster_overlay_data_uri(raster_scenario)
    raster_opacity = 0.68 if flood_scenario_selector != "show replacement costs only" else 0

    if selected_project == "Custom_AOI":
        custom_aoi = custom_aoi_to_gdf(custom_aoi_geojson)
        if custom_aoi is None:
            return (
                buildings_url, lcdb_data, EMPTY_GEOJSON, EMPTY_GEOJSON,
                raster_url, raster_opacity, "Draw an AOI to calculate its area and buffer.",
                None, None, None, None, None,
            )
        project_gdf_wgs84 = custom_aoi
        project_gdf_wgs84["id"] = project_gdf_wgs84.index.astype(str)
    else:
        project_gdf_wgs84 = project_layers[selected_project]["gdf"].copy()

    # Keep spatial calculations on projected geometries, independently of map display.
    project_gdf = project_gdf_wgs84.to_crs(epsg=2193)
    project_gdf_exploded = project_gdf.explode(index_parts=False).reset_index(drop=True)
    project_area_ha = project_gdf_exploded.geometry.area.sum() / 10000
    buffer_m = buffer_km
    project_gdf_exploded["buffer_geom"] = project_gdf_exploded.geometry.buffer(buffer_m)
    project_impact_area_ha = project_gdf_exploded["buffer_geom"].area.sum() / 10000

    gdf_buffer_wgs84 = project_gdf_exploded.set_geometry("buffer_geom").to_crs(epsg=4326)
    gdf_buffer_safe = gdf_buffer_wgs84.explode(index_parts=False).reset_index(drop=True)
    gdf_buffer_safe["id"] = gdf_buffer_safe.index.astype(str)
    gdf_buffer_safe["dummy"] = 0
    buffer_geojson = json.loads(
        gdf_buffer_safe[["id", "buffer_geom"]]
        .to_json(drop_id=True)
    )
    project_geojson = json.loads(project_gdf_wgs84.to_json(drop_id=True))

    buffer_union = project_gdf_exploded["buffer_geom"].union_all()
    buildings_in_buffer = gdf_buildings_projected[
        gdf_buildings_projected.geometry.within(buffer_union)
    ]
    n_houses = len(buildings_in_buffer)
    building_values = buildings_in_buffer["Rep_Cost"].sum()
    if flood_scenario_selector == "show replacement costs only":
        total_damage = 0
    else:
        flood_col = f"{flood_scenario_selector}_mean_flood"
        total_damage = (buildings_in_buffer["Rep_Cost"] * buildings_in_buffer[flood_col]).sum()

    summary = html.Div([
        html.Span(f"Houses in buffer: {n_houses:,}"),
        html.Span(f"Total value: NZ$ {building_values:,.0f}"),
        html.Span(f"Total damage: NZ$ {total_damage:,.0f}"),
    ], style={
        "display": "flex",
        "alignItems": "center",
        "flexWrap": "wrap",
        "gap": "8px 20px",
        "width": "100%",
    })

    return (
        buildings_url,
        lcdb_data,
        project_geojson,
        buffer_geojson,
        raster_url,
        raster_opacity,
        summary,
        project_area_ha,
        project_impact_area_ha,
        n_houses,
        building_values,
        total_damage,
    )

@app.callback(
    Output("ecosystem_bar_chart", "figure"),
    Input("intervention_type", "value")
)
def update_ecosystem_chart(selected_ecosystem):

    if selected_ecosystem is None:
        return go.Figure()

    # Get the dictionary for selected ecosystem
    services = ecosystem_services.get(selected_ecosystem, {})

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(services.keys()),
            y=list(services.values())
        )
    )

    fig.update_layout(
        title=f"Ecosystem Service Values – {selected_ecosystem}",
        xaxis_title="Ecosystem Services",
        yaxis_title="Value",
        template="plotly_white",
        xaxis_tickangle=-45,
        height=400,
        margin=dict(l=40, r=20, t=60, b=120)
    )

    return fig



@app.callback(
    Output("carbon_chart", "figure"),
    Output("carbon_table", "children"),
    Output("carbon_npv_output", "children"),
    Input("carbon_species", "value"),
    Input("carbon_price", "value"),
    Input("project_year", "value"),
    Input("project_area_ha", "value"),
    Input("discount_rate", "value"),

)
def update_carbon(selected_species, carbon_price, project_year,project_area_ha,discount_rate):
    if not selected_species or project_area_ha is None or project_area_ha <= 0:
        figure = go.Figure()
        figure.add_annotation(text="Select a species and a project area to view carbon results.")
        return figure, html.Div(""), ""
    if carbon_price is None:
        carbon_price = 0
    if discount_rate is None:
        discount_rate = 0.05
    if project_year is None:
        project_year = max(carbon_sequestration_data.keys())

    # --- Figure: original data (no interpolation) ---
    fig = go.Figure()
    for age_dict in carbon_sequestration_data.keys():
        pass  # placeholder, will fill below

    all_species = set()
    for age_dict in carbon_sequestration_data.values():
        all_species.update(age_dict.keys())

    species_series = {}  # will also be used for table interpolation

    for species in all_species:
        ages = []
        values = []
        for age in sorted(carbon_sequestration_data.keys()):
            val = carbon_sequestration_data[age].get(species, None)
            if val is not None:
                ages.append(age)
                values.append(val)
        species_series[species] = (ages, values)

        # Plot for figure
        if species == selected_species:
            fig.add_trace(go.Scatter(
                x=ages,
                y=values,
                mode="lines+markers",
                name=species.replace("_", " ").title(),
                line=dict(width=4),
                opacity=1
            ))
        else:
            fig.add_trace(go.Scatter(
                x=ages,
                y=values,
                mode="lines",
                name=species.replace("_", " ").title(),
                line=dict(width=1),
                opacity=0.25
            ))

    fig.update_layout(
        title=f"Carbon Sequestration Profiles (All Species)",
        xaxis_title="Age (years)",
        yaxis_title="tCO₂/ha/year",
        template="plotly_white",
        height=450
    )

    # --- Table: interpolate selected species annually up to project_year ---
    sel_ages_orig, sel_values_orig = species_series[selected_species]

    f = interp1d(sel_ages_orig, sel_values_orig, kind='linear', fill_value='extrapolate')
    interp_ages = np.arange(1, project_year + 1)
    interp_values = f(interp_ages)
    interp_values = np.maximum(f(interp_ages), 0)
    annual_value = interp_values * carbon_price
    # Total annual value ($/year) = annual_value * project area
    annual_value_total = annual_value * project_area_ha
    discount_factors = 1 / (1 + discount_rate) ** interp_ages
    #print(discount_factors)
    npv_values = annual_value_total * discount_factors
    cumulative_npv = np.cumsum(npv_values)
    # Final NPV
    carbon_npv = cumulative_npv[-1]
    # Final NP per ha 
    carbon_npv_per_ha = carbon_npv/project_area_ha 

    table = html.Div(
        style={
            "maxHeight": "300px",
            "overflowY": "auto",
            "border": "1px solid #ccc",
            "borderRadius": "5px",
            "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)",
            "marginTop": "20px"
        },
        children=[
            html.Table(
                style={"width": "100%", "borderCollapse": "collapse", "textAlign": "center"},
                children=[
                    html.Thead(
                        html.Tr([
                            html.Th("Year", style={"position": "sticky", "top": 0, "backgroundColor": "#4CAF50", "color": "white", "padding": "8px"}),
                            html.Th("tCO₂/ha/year", style={"position": "sticky", "top": 0, "backgroundColor": "#4CAF50", "color": "white", "padding": "8px"}),
                            html.Th("Carbon Price ($/tCO₂)", style={"position": "sticky", "top": 0, "backgroundColor": "#4CAF50", "color": "white", "padding": "8px"}),
                            html.Th("Annual Value ($/ha/year)", style={"position": "sticky", "top": 0, "backgroundColor": "#4CAF50", "color": "white", "padding": "8px"}),
                            html.Th("Annual Value ($/year)", style={"position": "sticky", "top": 0, "backgroundColor": "#4CAF50", "color": "white", "padding": "8px"}),
                            html.Th("Discounted Annual Value ($/year)", style={"position": "sticky", "top": 0, "backgroundColor": "#FF9800", "color": "white", "padding": "8px"}),
                            html.Th("Cumulative NPV ($/year)", style={"position": "sticky", "top": 0, "backgroundColor": "#FF9800", "color": "white", "padding": "8px"}),
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(int(age), style={"padding": "6px"}),
                            html.Td(f"{val:.2f}", style={"padding": "6px"}),
                            html.Td(f"{carbon_price:,.0f}", style={"padding": "6px"}),
                            html.Td(f"{ann_val:,.0f}", style={"padding": "6px"}),
                            html.Td(f"{ann_total:,.0f}", style={"padding": "6px"}),
                            html.Td(f"{disc_ann:,.0f}", style={"padding": "6px"}),
                            html.Td(f"{cum_npv:,.0f}", style={"padding": "6px"}),
                        ], style={"backgroundColor": "#f9f9f9"} if i % 2 == 0 else {"backgroundColor": "#ffffff"})
                        for i, (age, val, ann_val, ann_total,  disc_ann, cum_npv) in enumerate(
                            zip(interp_ages, interp_values, annual_value, annual_value_total, npv_values, cumulative_npv)
                        )
                    ])
                ]
            )
        ]
    )
    return fig, table, f"💰 Total Cumulative NPV of Carbon Sequestration over {project_year} years: ${carbon_npv:,.0f} for the entire project area or ${carbon_npv_per_ha:,.0f} on per ha basis"


@app.callback(
    Output("hazard_reduction_chart", "figure"),
    Output("hazard_reduction_table", "children"),
    Output("hazard_npv_summary", "children"),
    Input("intervention_type", "value"),
    Input("carbon_species", "value"),
    Input("hazard_type", "value"),
    Input("building_values", "value"),
    Input("total_damage", "value"),
    Input("project_year", "value"),
    Input("discount_rate", "value"),
    Input("project_impact_area_ha", "value"),
)
def update_hazard_reduction_summary(intervention, carbon_species, selected_hazards, building_values, total_damage, project_year, discount_rate, project_impact_area_ha):
    if not selected_hazards or not intervention:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Select intervention and hazards to see results.")
        return empty_fig, html.Div(""), html.Div("")
    
    if project_year is None:
        project_year = max(tree_growth_curves.keys())
    
    if discount_rate is None:
        discount_rate = 0.05
    
    max_reduction_map = default_nbs_reduction.get(intervention, {})
    
    # Loop over hazards and collect all data
    all_rows = []
    for hazard_type in selected_hazards:
        species = carbon_species
        timeseries = hazard_timeseries(carbon_species, tree_growth_curves, max_reduction_map, hazard_type)
        for year, row in timeseries.items():
            all_rows.append(row)
    
    all_hazard_timeseries = pd.DataFrame(all_rows)
    
    # Create figure showing hazard reduction % over time (up to 120 years)
    fig = go.Figure()
    for hazard_type in selected_hazards:
        filtered_df = all_hazard_timeseries[all_hazard_timeseries["hazard_type"] == hazard_type]
        ages = filtered_df["year"]
        values = filtered_df["hazard_reduction_pct"]
        fig.add_trace(go.Scatter(
            x=ages,
            y=values,
            mode="lines",
            name=f"{species.replace('_', ' ').title()} - {hazard_map.get(hazard_type)}",
            line=dict(width=2)
        ))

    fig.update_layout(
        title=f"Hazard Reduction Profiles for {species.replace('_', ' ').title()}",
        xaxis_title="Age (years)",
        yaxis_title="Hazard Reduction (%)",
        template="plotly_white",
        height=450
    )

    # Filter table data to project year only
    table_hazard_timeseries = all_hazard_timeseries[all_hazard_timeseries["year"] <= project_year]

    # Create table showing property value protected per hazard type per year
    if total_damage is None or total_damage <= 0:
        table_html = html.Div("No baseline damage in impact area to calculate damage averted.")
        npv_summary = html.Div("")
    else:
        # Pivot table: years as rows, hazard types as columns
        pivot_data = {}
        for hazard_type in selected_hazards:
            filtered_df = table_hazard_timeseries[table_hazard_timeseries["hazard_type"] == hazard_type]
            for idx, row in filtered_df.iterrows():
                year = row["year"]
                if year not in pivot_data:
                    pivot_data[year] = {}
                hazard_reduction_pct = row["hazard_reduction_pct"]
                protected_value = total_damage * (hazard_reduction_pct / 100) *0.01 #probablity of flooding
                discount_factor = 1 / (1 + discount_rate) ** year
                discounted_value = protected_value * discount_factor
                pivot_data[year][hazard_type] = {
                    "reduction_pct": hazard_reduction_pct,
                    "protected_value": protected_value,
                    "discounted_value": discounted_value
                }
        
        # Color scheme for hazard types
        hazard_colors = {
            "Flood": "#E3F2FD",
            "River_Flood": "#F3E5F5",
            "Tsunami": "#FCE4EC",
            "Erosion": "#FFF3E0",
            "Landslide": "#E0F2F1"
        }
        
        # Create table header with hazard types as columns with color blocks
        header_cells = [html.Th("Year", style={"position": "sticky", "top": 0, "backgroundColor": "#2196F3", "color": "white", "padding": "8px"})]
        header_cells.append(html.Th("Total Baseline Damage (NZ$)", style={"position": "sticky", "top": 0, "backgroundColor": "#2196F3", "color": "white", "padding": "8px"}))
        
        for hazard_type in selected_hazards:
            hazard_color = hazard_colors.get(hazard_type, "#EEEEEE")
            header_cells.append(html.Th(
                f"{hazard_map.get(hazard_type, hazard_type)}\nReduction (%)",
                style={"position": "sticky", "top": 0, "backgroundColor": hazard_color, "color": "#333", "padding": "8px", "whiteSpace": "pre-wrap", "fontWeight": "bold", "borderLeft": "3px solid #666"}
            ))
            header_cells.append(html.Th(
                f"{hazard_map.get(hazard_type, hazard_type)}\nDamage Averted (NZ$)",
                style={"position": "sticky", "top": 0, "backgroundColor": hazard_color, "color": "#333", "padding": "8px", "whiteSpace": "pre-wrap", "fontWeight": "bold"}
            ))
            header_cells.append(html.Th(
                f"{hazard_map.get(hazard_type, hazard_type)}\nDiscounted Damage Averted (NZ$)",
                style={"position": "sticky", "top": 0, "backgroundColor": hazard_color, "color": "#333", "padding": "8px", "whiteSpace": "pre-wrap", "fontWeight": "bold", "borderRight": "3px solid #666"}
            ))
        
        header_cells.append(html.Th("Cumulative NPV (NZ$)", style={"position": "sticky", "top": 0, "backgroundColor": "#FF9800", "color": "white", "padding": "8px"}))
        
        # Create table rows with NPV calculations
        table_rows = []
        cumulative_npv_total = 0
        years_sorted = sorted(pivot_data.keys())
        
        for i, year in enumerate(years_sorted):
            row_cells = [html.Td(int(year), style={"padding": "6px", "fontWeight": "bold"})]
            row_cells.append(html.Td(f"${total_damage:,.0f}", style={"padding": "6px"}))
            
            year_total_discounted = 0
            for hazard_type in selected_hazards:
                hazard_color = hazard_colors.get(hazard_type, "#EEEEEE")
                if hazard_type in pivot_data[year]:
                    reduction_pct = pivot_data[year][hazard_type]["reduction_pct"]
                    protected_value = pivot_data[year][hazard_type]["protected_value"]
                    discounted_value = pivot_data[year][hazard_type]["discounted_value"]
                    year_total_discounted += discounted_value
                    
                    row_cells.append(html.Td(f"{reduction_pct:.2f}", style={"padding": "6px", "backgroundColor": hazard_color, "borderLeft": "3px solid #999"}))
                    row_cells.append(html.Td(f"${protected_value:,.0f}", style={"padding": "6px", "fontWeight": "bold", "color": "#27ae60", "backgroundColor": hazard_color}))
                    row_cells.append(html.Td(f"${discounted_value:,.0f}", style={"padding": "6px", "fontWeight": "bold", "color": "#FF9800", "backgroundColor": hazard_color, "borderRight": "3px solid #999"}))
                else:
                    row_cells.append(html.Td("-", style={"padding": "6px", "backgroundColor": hazard_color, "borderLeft": "3px solid #999"}))
                    row_cells.append(html.Td("-", style={"padding": "6px", "backgroundColor": hazard_color}))
                    row_cells.append(html.Td("-", style={"padding": "6px", "backgroundColor": hazard_color, "borderRight": "3px solid #999"}))
            
            cumulative_npv_total += year_total_discounted
            row_cells.append(html.Td(f"${cumulative_npv_total:,.0f}", style={"padding": "6px", "fontWeight": "bold", "color": "#FF9800"}))
            
            bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
            table_rows.append(html.Tr(row_cells, style={"backgroundColor": bg_color}))
        
        table_html = html.Div(
            style={
                "maxHeight": "400px",
                "overflowY": "auto",
                "overflowX": "auto",
                "border": "1px solid #ccc",
                "borderRadius": "5px",
                "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)",
                "marginTop": "20px"
            },
            children=[
                html.Table(
                    style={"width": "100%", "borderCollapse": "collapse", "textAlign": "center", "minWidth": "600px"},
                    children=[
                        html.Thead(html.Tr(header_cells)),
                        html.Tbody(table_rows)
                    ]
                )
            ]
        )
        
        # Calculate NPV per ha
        npv_per_ha = cumulative_npv_total / project_impact_area_ha if project_impact_area_ha and project_impact_area_ha > 0 else 0
        
        npv_summary = html.Div([
            html.P(
                f"💰 Annualised baseline damage for 1-in-100 years flood event is  ${total_damage*0.01:,.0f} for the entire project impact area (adjust for 1% annualised probability) basis",
                style={"marginTop": "15px", "fontSize": "20px", "fontWeight": "bold", "color": "#030303"}
            ),
            html.P(
                f"💰 Total Cumulative NPV of Hazard Reduction over {project_year} years: ${cumulative_npv_total:,.0f} for the entire project area or ${npv_per_ha:,.0f} per ha basis",
                style={"marginTop": "5px", "fontSize": "20px", "fontWeight": "bold", "color": "#030303"}
            )
        ], style={"marginTop": "10px"})
    
    return fig, table_html, npv_summary

@app.callback(
    Output("establishment_cost_input", "value"),
    Output("maintenance_cost_input", "value"),
    Input("intervention_type", "value")
)
def update_cost_inputs(intervention):
    if intervention is None:
        establishment_base = 100000
        maintenance_base = 1000
    else:
        establishment_base = establishment_costs.get(intervention, 100000)
        maintenance_base = maintenance_costs.get(intervention, 1000)

    return (
        round(establishment_base, 2),
        round(maintenance_base, 2)
    )

@app.callback(
    Output("cost_chart", "figure"),
    Output("cost_summary", "children"),
    Input("intervention_type", "value"),
    Input("project_year", "value"),
    Input("project_area_ha", "value"),
    Input("discount_rate", "value"),
    Input("establishment_cost_input", "value"),
    Input("maintenance_cost_input", "value"),
)
def update_cost_chart(
    intervention,
    project_year,
    project_area_ha,
    discount_rate,
    establishment_cost_per_ha,
    maintenance_cost_per_ha,
):
    if not intervention or not project_year or not project_area_ha:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Select intervention and define project parameters to see costs.")
        empty_summary = html.Div("No data available")
        return empty_fig, empty_summary
    
    if discount_rate is None:
        discount_rate = 0.05
    if establishment_cost_per_ha is None:
        establishment_cost_per_ha = 100000
    if maintenance_cost_per_ha is None:
        maintenance_cost_per_ha = 1000

    # Calculate total establishment cost
    # Use project area for consistency with NPV summary
    area_for_cost = project_area_ha
    total_establishment_cost = establishment_cost_per_ha * area_for_cost
    
    # Calculate annual costs and NPV
    years = np.arange(1, project_year + 1)
    annual_costs = np.ones(project_year) * maintenance_cost_per_ha * area_for_cost
    discount_factors = 1 / (1 + discount_rate) ** years
    discounted_costs = annual_costs * discount_factors
    cumulative_maintenance_series = np.cumsum(discounted_costs)
    cumulative_maintenance_cost = np.sum(discounted_costs)
    
    # Total life cycle cost
    total_lifecycle_cost = total_establishment_cost + cumulative_maintenance_cost
    
    # Create stacked area chart
    fig = go.Figure()
    
    # Add establishment cost as horizontal line
    fig.add_trace(go.Scatter(
        x=years,
        y=[total_establishment_cost] * project_year,
        mode="lines",
        name="Establishment Cost (one-time)",
        line=dict(width=2, color="#FF6B6B"),
        fill=None
    ))
    
    # Add discounted annual maintenance costs
    fig.add_trace(go.Scatter(
        x=years,
        y=cumulative_maintenance_series,
        mode="lines+markers",
        name="Cumulative Maintenance Cost (discounted)",
        line=dict(width=2, color="#4ECDC4"),
        fill="tozeroy"
    ))
    
    # Add total life cycle cost
    fig.add_trace(go.Scatter(
        x=years,
        y=total_establishment_cost + cumulative_maintenance_series,
        mode="lines+markers",
        name="Total Life Cycle Cost",
        line=dict(width=3, color="#2C3E50", dash="dash"),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"Life Cycle Cost Analysis for {intervention.replace('_',' ')}",
        xaxis_title="Year",
        yaxis_title="Cost (NZ$)",
        template="plotly_white",
        height=500,
        hovermode='x unified',
        yaxis=dict(tickformat="$,.0f")
    )
    
    # Create cost summary - all in one line
    summary = html.Div([
        html.Div([
            html.Span(f"📍 {project_area_ha:.2f} ha | ", style={"marginRight": "20px", "fontWeight": "bold", "fontSize": "14px"}),
            html.Span(f"🏗️ Total Est.: NZ$ {total_establishment_cost:,.0f} | ", style={"marginRight": "20px", "fontWeight": "bold", "fontSize": "14px", "color": "#010101"}),
            html.Span(f"💰 Maint. ({project_year}yr): NZ$ {cumulative_maintenance_cost:,.0f} | ", style={"marginRight": "20px", "fontWeight": "bold", "fontSize": "14px", "color": "#030303"}),
            html.Span(f"📊 Total LCC: NZ$ {total_lifecycle_cost:,.0f}", style={"fontWeight": "bold", "fontSize": "14px", "color": "#DF0808", "backgroundColor": "#ffffff", "padding": "5px 10px", "borderRadius": "3px"})
        ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "10px", "padding": "15px", "backgroundColor": "#f9f9f9", "borderRadius": "5px", "border": "1px solid #ddd"})
    ])
    
    return fig, summary

@app.callback(
    Output("npv_waterfall", "figure"),
    Input("intervention_type", "value"),
    Input("carbon_species", "value"),
    Input("hazard_type", "value"),
    Input("project_year", "value"),
    Input("carbon_price", "value"),
    Input("project_area_ha", "value"),
    Input("project_impact_area_ha", "value"),
    Input("total_damage", "value"),
    Input("discount_rate", "value"),
    Input("establishment_cost_input", "value"),
    Input("maintenance_cost_input", "value"),
)
def update_npv_waterfall(
    intervention,
    carbon_species,
    selected_hazards,
    project_year,
    carbon_price,
    project_area_ha,
    project_impact_area_ha,
    total_damage,
    discount_rate,
    establishment_cost_per_ha,
    maintenance_cost_per_ha,
):
    if not intervention or not carbon_species or not selected_hazards or not project_year or not project_area_ha:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Select all parameters to view NPV Waterfall.")
        return empty_fig
    
    if carbon_price is None:
        carbon_price = 0
    if discount_rate is None:
        discount_rate = 0.05
    if total_damage is None:
        total_damage = 0
    if establishment_cost_per_ha is None:
        establishment_cost_per_ha = 100000
    if maintenance_cost_per_ha is None:
        maintenance_cost_per_ha = 1000
    
    total_establishment_cost = establishment_cost_per_ha * project_area_ha
    years = np.arange(1, project_year + 1)
    annual_costs = np.ones(project_year) * maintenance_cost_per_ha * project_area_ha
    discount_factors = 1 / (1 + discount_rate) ** years
    discounted_costs = annual_costs * discount_factors
    cumulative_maintenance_cost = np.sum(discounted_costs)
    total_costs = total_establishment_cost + cumulative_maintenance_cost
    
    # 2. CARBON BENEFITS
    sel_ages_orig, sel_values_orig = [], []
    for age in sorted(carbon_sequestration_data.keys()):
        val = carbon_sequestration_data[age].get(carbon_species, None)
        if val is not None:
            sel_ages_orig.append(age)
            sel_values_orig.append(val)
    
    if sel_ages_orig and project_area_ha:
        f = interp1d(sel_ages_orig, sel_values_orig, kind='linear', fill_value='extrapolate')
        interp_ages = np.arange(1, project_year + 1)
        interp_values = np.maximum(f(interp_ages), 0)
        annual_value = interp_values * carbon_price
        annual_value_total = annual_value * project_area_ha
        discount_factors_carbon = 1 / (1 + discount_rate) ** interp_ages
        npv_values = annual_value_total * discount_factors_carbon
        carbon_npv = np.sum(npv_values)
    else:
        carbon_npv = 0
    
    # 3. HAZARD REDUCTION BENEFITS
    max_reduction_map = default_nbs_reduction.get(intervention, {})
    all_rows = []
    for hazard_type in selected_hazards:
        timeseries = hazard_timeseries(carbon_species, tree_growth_curves, max_reduction_map, hazard_type)
        for year, row in timeseries.items():
            all_rows.append(row)
    
    if all_rows and total_damage > 0:
        all_hazard_timeseries = pd.DataFrame(all_rows)
        table_hazard_timeseries = all_hazard_timeseries[all_hazard_timeseries["year"] <= project_year]
        
        hazard_npv_total = 0
        for hazard_type in selected_hazards:
            filtered_df = table_hazard_timeseries[table_hazard_timeseries["hazard_type"] == hazard_type]
            for idx, row in filtered_df.iterrows():
                year = row["year"]
                hazard_reduction_pct = row["hazard_reduction_pct"]
                protected_value = total_damage * (hazard_reduction_pct / 100) * 0.01 #probability of flooding
                discount_factor = 1 / (1 + discount_rate) ** year
                discounted_value = protected_value * discount_factor
                hazard_npv_total += discounted_value
    else:
        hazard_npv_total = 0
    
    # Calculate total NPV
    total_npv = carbon_npv + hazard_npv_total - total_costs
    
    # Create waterfall chart
    categories = [
        "Starting Point",
        "Establishment Cost",
        "Maintenance Cost",
        "Carbon Benefits",
        "Hazard Benefits",
        "Total NPV"
    ]
    
    values = [
        0,
        -total_establishment_cost,
        -cumulative_maintenance_cost,
        carbon_npv,
        hazard_npv_total,
        total_npv
    ]
    
    # Create measure array for waterfall
    measure = ["absolute", "relative", "relative", "relative", "relative", "total"]
    
    # Text for annotations
    text_vals = [
        "Start",
        f"${total_establishment_cost:,.0f}",
        f"${cumulative_maintenance_cost:,.0f}",
        f"${carbon_npv:,.0f}",
        f"${hazard_npv_total:,.0f}",
        f"${total_npv:,.0f}"
    ]
    
    fig = go.Figure(go.Waterfall(
        x=categories,
        y=values,
        measure=measure,
        text=text_vals,
        textposition="outside",
        connector=dict(line=dict(color="#666")),
        increasing=dict(marker=dict(color="#4ECDC4")),
        decreasing=dict(marker=dict(color="#FF6B6B")),
        totals=dict(marker=dict(color="#2C3E50"))
    ))
    
    fig.update_layout(
        title=f"NPV Waterfall Analysis: {intervention} - {carbon_species.replace('_', ' ').title()}",
        yaxis_title="Net Present Value (NZ$)",
        template="plotly_white",
        height=600,
        yaxis=dict(tickformat="$,.0f"),
        showlegend=False
    )
    
    return fig

if __name__ == "__main__":
    app.run(debug=True)