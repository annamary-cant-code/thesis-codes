import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
import pyproj


# --- choose file depending on bounding box in analysis ---
# OSM_FILE = "map_test.osm"
OSM_FILE = "map_full.osm"

# --- Parse XML ---
tree = ET.parse(OSM_FILE)
root = tree.getroot()

# --- Build node coordinate lookup ---
nodes = {}
for node in root.findall("node"):
    nid = node.get("id")
    lat = float(node.get("lat"))
    lon = float(node.get("lon"))
    nodes[nid] = (lon, lat)   # (x, y) convention

# --- Tag sets ---
# Buildings — all values present in map_full.osm worth classifying as built structure
BUILDING_VALS = {
    "yes", "apartments", "residential", "house", "garage", "school",
    "roof", "church", "garages", "industrial", "retail", "detached",
    "commercial", "office", "university", "hospital", "hotel", "public",
    "government", "chapel", "kindergarten", "dormitory", "warehouse",
    "train_station", "tower", "supermarket", "sports_centre", "college",
    "villa", "theatre", "cinema", "cathedral", "castle", "stadium"
}

# Roads — highway values present in map_full.osm
HIGHWAY_VALS = {
    "footway", "residential", "service", "tertiary", "secondary",
    "cycleway", "pedestrian", "primary", "steps", "path",
    "unclassified", "busway", "track", "tertiary_link", "corridor",
    "primary_link", "living_street", "secondary_link"
}

# Railways — active lines only (not disused/razed/abandoned)
RAILWAY_VALS = {
    "tram", "rail", "subway"
}

# Green/natural — landuse values that represent vegetation or open land
LANDUSE_GREEN_VALS = {
    "grass", "farmland", "flowerbed", "meadow", "recreation_ground",
    "forest", "allotments", "vineyard", "orchard", "greenfield"
}

# Green/natural — natural values that represent vegetation
NATURAL_GREEN_VALS = {
    "wood", "scrub", "grassland"
}

# Green/natural — leisure values that represent open green spaces
LEISURE_GREEN_VALS = {
    "garden", "pitch", "park", "playground", "dog_park",
    "track", "schoolyard", "nature_reserve"
}


# --- Helper: extract node coordinates for a <way> ---
def way_coords(way):
    return [nodes[nd.get("ref")]
            for nd in way.findall("nd")
            if nd.get("ref") in nodes]

# --- Helper: extract tags from any element ---
def get_tags(el):
    return {t.get("k"): t.get("v") for t in el.findall("tag")}

# --- Collect geometries per class ---
water_geoms    = []
building_geoms = []
road_geoms     = []
park_geoms     = []

for way in root.findall("way"):
    tags   = get_tags(way)
    coords = way_coords(way)
    if len(coords) < 2:
        continue

    is_closed = coords[0] == coords[-1]

    # WATER — waterway tag present
    if "waterway" in tags:
        name = tags.get("name", "")
        if is_closed and len(coords) >= 4:
            water_geoms.append(Polygon(coords))
        else:
            if "Dora" in name:
                buf = 0.0002   # Dora Riparia — ~18 m
            else:
                buf = 0.0001   # small streams
            water_geoms.append(LineString(coords).buffer(buf))

    # BUILDING
    elif tags.get("building") in BUILDING_VALS:
        if is_closed and len(coords) >= 4:
            building_geoms.append(Polygon(coords))

    # ROAD — highway
    elif tags.get("highway") in HIGHWAY_VALS:
        road_geoms.append(LineString(coords))

    # ROAD — railway (active lines only)
    elif tags.get("railway") in RAILWAY_VALS:
        road_geoms.append(LineString(coords))

    # GREEN — landuse (vegetation and open land)
    elif tags.get("landuse") in LANDUSE_GREEN_VALS:
        if is_closed and len(coords) >= 4:
            park_geoms.append(Polygon(coords))

    # GREEN — natural (vegetation)
    elif tags.get("natural") in NATURAL_GREEN_VALS:
        if is_closed and len(coords) >= 4:
            park_geoms.append(Polygon(coords))

    # GREEN — leisure (open green spaces)
    elif tags.get("leisure") in LEISURE_GREEN_VALS:
        if is_closed and len(coords) >= 4:
            park_geoms.append(Polygon(coords))

# --- Also grab waterway relations (the Po river) ---
# Build a way_id -> coords lookup for relation assembly
way_coords_map = {}
for way in root.findall("way"):
    coords = way_coords(way)
    if len(coords) >= 2:
        way_coords_map[way.get("id")] = coords

for rel in root.findall("relation"):
    tags = get_tags(rel)
    if tags.get("type") == "waterway":
        name = tags.get("name", "")
        for member in rel.findall("member"):
            if member.get("type") == "way":
                wid = member.get("ref")
                if wid in way_coords_map:
                    coords = way_coords_map[wid]
                    if len(coords) >= 2:
                        if "Dora" in name:
                            buf = 0.0002   # Dora Riparia
                        else:
                            buf = 0.0007   # Po and other large rivers
                        water_geoms.append(LineString(coords).buffer(buf))

# --- Merge into single geometries ---
water_union    = unary_union(water_geoms)    if water_geoms    else None
building_union = unary_union(building_geoms) if building_geoms else None
road_union     = unary_union(road_geoms)     if road_geoms     else None
park_union     = unary_union(park_geoms)     if park_geoms     else None

print(f"Water geometries:    {len(water_geoms)}")
print(f"Building geometries: {len(building_geoms)}")
print(f"Road geometries:     {len(road_geoms)}")
print(f"Green geometries:    {len(park_geoms)}")



# =============================================================================
# BLOCK 2 — Coordinate conversion and grid setup
# =============================================================================

import pyproj
import numpy as np

# --- GRID RESOLUTION (only parameter to change for coarser/finer runs) ---
N_ROWS = 500
N_COLS = 500

# --- Extract bounding box directly from the already-parsed XML root ---
bounds_el = root.find("bounds")
LAT_MIN = float(bounds_el.get("minlat"))
LAT_MAX = float(bounds_el.get("maxlat"))
LON_MIN = float(bounds_el.get("minlon"))
LON_MAX = float(bounds_el.get("maxlon"))
print(f"Bounding box: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")

# --- Coordinate transformer: WGS84 (degrees) -> UTM zone 32N (metres) ---
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

def to_utm_point(lon, lat):
    return transformer.transform(lon, lat)

def to_utm_geometry(geom):
    from shapely.ops import transform
    return transform(transformer.transform, geom)

# --- Reproject all terrain unions to UTM ---
water_utm    = to_utm_geometry(water_union)   if water_union    else None
building_utm = to_utm_geometry(building_union) if building_union else None
road_utm     = to_utm_geometry(road_union)     if road_union     else None
park_utm     = to_utm_geometry(park_union)     if park_union     else None

# --- Bounding box corners in metres ---
x_min, y_min = to_utm_point(LON_MIN, LAT_MIN)
x_max, y_max = to_utm_point(LON_MAX, LAT_MAX)

cell_w = (x_max - x_min) / N_COLS
cell_h = (y_max - y_min) / N_ROWS
print(f"Cell size: {cell_w:.1f} m x {cell_h:.1f} m")

# --- Grid of cell centre coordinates in metres ---
xs = np.linspace(x_min + cell_w/2, x_max - cell_w/2, N_COLS)
ys = np.linspace(y_min + cell_h/2, y_max - cell_h/2, N_ROWS)
grid_x, grid_y = np.meshgrid(xs, ys)   # shape (N_ROWS, N_COLS)

print(f"Grid shape: {grid_x.shape}")

# =============================================================================
# BLOCK 3 — Classify every grid cell
# =============================================================================

from shapely.geometry import Point
from shapely.strtree import STRtree

# --- Road buffer in metres (gives roads physical width on the grid) ---
ROAD_BUFFER_M = 5.0

# --- Label encoding ---
GROUND   = 0
PARK     = 1
ROAD     = 2
WATER    = 3
BUILDING = 4

# --- Build spatial indices for fast lookup ---
# STRtree expects a list of geometries — extract them from each union
def build_tree(union):
    if union is None:
        return None, []
    if union.geom_type == "GeometryCollection" or union.geom_type.startswith("Multi"):
        geoms = list(union.geoms)
    else:
        geoms = [union]
    return STRtree(geoms), geoms

water_tree,    _ = build_tree(water_utm)
building_tree, _ = build_tree(building_utm)
park_tree,     _ = build_tree(park_utm)

# Roads: pre-buffer the entire union once (much faster than buffering per point)
road_buffered = road_utm.buffer(ROAD_BUFFER_M) if road_utm else None
road_tree,   _ = build_tree(road_buffered)

# --- Classification loop ---
terrain_grid = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)

for i in range(N_ROWS):
    for j in range(N_COLS):
        pt = Point(grid_x[i, j], grid_y[i, j])

        # Priority order: park < road < water < building (higher overwrites lower)
        if park_tree     and park_tree.query(pt,     predicate="intersects").size > 0:
            terrain_grid[i, j] = PARK
        if road_tree     and road_tree.query(pt,     predicate="intersects").size > 0:
            terrain_grid[i, j] = ROAD
        if water_tree    and water_tree.query(pt,    predicate="intersects").size > 0:
            terrain_grid[i, j] = WATER
        if building_tree and building_tree.query(pt, predicate="intersects").size > 0:
            terrain_grid[i, j] = BUILDING

print("Classification complete.")
print(f"  Ground:   {(terrain_grid == GROUND).sum():6d} cells ({100*(terrain_grid == GROUND).mean():.1f}%)")
print(f"  Park:     {(terrain_grid == PARK).sum():6d} cells ({100*(terrain_grid == PARK).mean():.1f}%)")
print(f"  Road:     {(terrain_grid == ROAD).sum():6d} cells ({100*(terrain_grid == ROAD).mean():.1f}%)")
print(f"  Water:    {(terrain_grid == WATER).sum():6d} cells ({100*(terrain_grid == WATER).mean():.1f}%)")
print(f"  Building: {(terrain_grid == BUILDING).sum():6d} cells ({100*(terrain_grid == BUILDING).mean():.1f}%)")


# =============================================================================
# BLOCK 4 — Visualise and save
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pickle

# --- Colormap matching terrain intuition ---
cmap = mcolors.ListedColormap([
    "#d4b97a",   # 0 ground   — sandy beige
    "#6db56d",   # 1 park     — green
    "#999999",   # 2 road     — grey
    "#4a90d9",   # 3 water    — blue
    "#c0392b",   # 4 building — red
])
bounds_cmap = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds_cmap, cmap.N)

fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(
    terrain_grid,
    origin="lower",        # y=0 at bottom (south), geographic convention
    extent=[x_min, x_max, y_min, y_max],
    cmap=cmap,
    norm=norm,
    interpolation="nearest"
)

# --- Legend ---
legend_entries = [
    mpatches.Patch(color="#d4b97a", label="Ground"),
    mpatches.Patch(color="#6db56d", label="Green/Natural"),
    mpatches.Patch(color="#999999", label="Road"),
    mpatches.Patch(color="#4a90d9", label="Water"),
    mpatches.Patch(color="#c0392b", label="Building"),
]
ax.legend(handles=legend_entries, loc="upper right", fontsize=9)

ax.set_xlabel("Easting (m, UTM 32N)")
ax.set_ylabel("Northing (m, UTM 32N)")
ax.set_title("Terrain classification — Phase 1 output")
plt.tight_layout()
plt.savefig("TERRAIN_MAP.png", dpi=150)
plt.show()
print("Saved terrain_map.png")

# --- Save everything needed by later phases ---
with open("terrain_stage1.pkl", "wb") as f:
    pickle.dump({
        "terrain_grid": terrain_grid,
        "grid_x":       grid_x,
        "grid_y":       grid_y,
        "x_min":  x_min,  "x_max": x_max,
        "y_min":  y_min,  "y_max": y_max,
        "cell_w": cell_w, "cell_h": cell_h,
        "N_ROWS": N_ROWS, "N_COLS": N_COLS,
        "LAT_MIN": LAT_MIN, "LAT_MAX": LAT_MAX,
        "LON_MIN": LON_MIN, "LON_MAX": LON_MAX,
    }, f)
print("Saved terrain_stage1.pkl")