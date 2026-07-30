# =============================================================================
# PHASE 2 — Trajectory definition, spline interpolation, visualisation
# =============================================================================

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.interpolate import splprep, splev
import pyproj

# --- Load Phase 1 outputs ---
with open("terrain_stage1.pkl", "rb") as f:
    data = pickle.load(f)

terrain_grid = data["terrain_grid"]
grid_x       = data["grid_x"]
grid_y       = data["grid_y"]
x_min        = data["x_min"];  x_max = data["x_max"]
y_min        = data["y_min"];  y_max = data["y_max"]
LAT_MIN      = data["LAT_MIN"]; LAT_MAX = data["LAT_MAX"]
LON_MIN      = data["LON_MIN"]; LON_MAX = data["LON_MAX"]

# =============================================================================
# BLOCK 1 — Load waypoints from Excel and validate
# =============================================================================

# WAYPOINTS_FILE   = "waypoints_PO-e-STURA.xlsx"
# WAYPOINTS_FILE   = "waypoints_SOLO-PO.xlsx"
WAYPOINTS_FILE   = "waypoints_STRAIGHT-LINE.xlsx"
SAMPLE_SPACING_M = 10.0   # metres between trajectory sample points

# Coordinate transformer (same as Phase 1)
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

def latlon_to_utm(lat, lon):
    return transformer.transform(lon, lat)

# Load waypoints
wp = pd.read_excel(WAYPOINTS_FILE)
print(f"Loaded {len(wp)} waypoints from {WAYPOINTS_FILE}")

# Validate: drop any waypoint outside the bounding box
valid_mask = (
    (wp["lat"] >= LAT_MIN) & (wp["lat"] <= LAT_MAX) &
    (wp["lon"] >= LON_MIN) & (wp["lon"] <= LON_MAX)
)
dropped = wp[~valid_mask]
if len(dropped) > 0:
    print(f"WARNING: {len(dropped)} waypoint(s) outside bounding box and ignored:")
    for _, row in dropped.iterrows():
        print(f"  {row.get('name', '?')}  lat={row['lat']}  lon={row['lon']}")

wp = wp[valid_mask].reset_index(drop=True)
print(f"Using {len(wp)} valid waypoints.")

if len(wp) < 2:
    raise ValueError("Need at least 2 valid waypoints to define a trajectory.")

# Convert waypoints to UTM metres
wp_x = np.array([latlon_to_utm(row["lat"], row["lon"])[0] for _, row in wp.iterrows()])
wp_y = np.array([latlon_to_utm(row["lat"], row["lon"])[1] for _, row in wp.iterrows()])

print(f"Waypoints in UTM (metres):")
for i, row in wp.iterrows():
    print(f"  {str(row.get('name','')):<20s}  x={wp_x[i]:.1f}  y={wp_y[i]:.1f}")

# =============================================================================
# BLOCK 2 — Fit spline and sample at fixed physical spacing
# =============================================================================

# Fit parametric spline through waypoints
# k=3 (cubic) requires at least 4 points; fall back to k=1 (linear) if fewer
k = min(3, len(wp) - 1)
tck, u = splprep([wp_x, wp_y], s=0, k=k)

# Estimate total arc length with a dense initial sample
u_dense   = np.linspace(0, 1, 100000)
x_dense, y_dense = splev(u_dense, tck)
dx = np.diff(x_dense)
dy = np.diff(y_dense)
seg_lengths    = np.sqrt(dx**2 + dy**2)
cumulative_len = np.concatenate([[0], np.cumsum(seg_lengths)])
total_length_m = cumulative_len[-1]
print(f"\nTotal trajectory length: {total_length_m:.1f} m")

# Re-parameterise uniformly by arc length
n_samples  = int(np.floor(total_length_m / SAMPLE_SPACING_M)) + 1
target_distances = np.linspace(0, total_length_m, n_samples)
u_uniform  = np.interp(target_distances, cumulative_len, u_dense)

# Sample the spline at uniform spacing
traj_x, traj_y = splev(u_uniform, tck)
traj_x = np.array(traj_x)
traj_y = np.array(traj_y)

print(f"Trajectory sampled at every {SAMPLE_SPACING_M} m: {n_samples} points total")

# --- Compute heading angle at each sample point (radians from East) ---
# Central differences for interior points, forward/backward for endpoints
dx = np.gradient(traj_x)
dy = np.gradient(traj_y)
traj_theta_deg = np.degrees(np.arctan2(dy, dx))   # (n_samples,) in degrees

print(f"Heading range: {traj_theta_deg.min():.1f}° to {traj_theta_deg.max():.1f}°")


# --- Diagnostic: print first 20 entries of the cumulative distance table ---
print("\nDiagnostic: first 20 entries of cumulative distance table")
print(f"{'Index':>8}  {'t value':>10}  {'x (m)':>12}  {'y (m)':>12}  {'cumul. dist (m)':>16}")
for idx in range(20):
    print(f"  {idx:6d}  {u_dense[idx]:10.5f}  {x_dense[idx]:12.2f}  {y_dense[idx]:12.2f}  {cumulative_len[idx]:16.4f}")

print(f"\n  ...")
print(f"  Total arc length: {total_length_m:.4f} m")
print(f"  Average spacing between t points: {total_length_m / len(u_dense):.4f} m")
print(f"  Requested sample spacing: {SAMPLE_SPACING_M} m")
print(f"  Resolution ratio (sample/t-spacing): {SAMPLE_SPACING_M / (total_length_m / len(u_dense)):.1f}x  ← should be >> 1")

# --- Diagnostic: print first 10 uniform sample points ---
print(f"\nDiagnostic: first 10 uniform sample points (every {SAMPLE_SPACING_M} m)")
print(f"{'Index':>8}  {'t value':>10}  {'x (m)':>12}  {'y (m)':>12}  {'target dist (m)':>16}")
for idx in range(10):
    print(f"  {idx:6d}  {u_uniform[idx]:10.6f}  {traj_x[idx]:12.2f}  {traj_y[idx]:12.2f}  {target_distances[idx]:16.4f}")

# =============================================================================
# BLOCK 3 — Visualise trajectory overlaid on terrain map
# =============================================================================

cmap = mcolors.ListedColormap([
    "#d4b97a",   # 0 ground
    "#6db56d",   # 1 park
    "#999999",   # 2 road
    "#4a90d9",   # 3 water
    "#c0392b",   # 4 building
])
bounds_cmap = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds_cmap, cmap.N)

fig, ax = plt.subplots(figsize=(10, 8))

# Terrain base map
ax.imshow(
    terrain_grid,
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
    alpha=0.85
)

# Trajectory line
ax.plot(traj_x, traj_y,
        color="black", linewidth=1.5,
        label="Trajectory", zorder=3)

# Dots at waypoints only
ax.scatter(wp_x, wp_y,
           color="yellow", edgecolors="black",
           s=40, zorder=5, label="Waypoints")

# Waypoint labels
for i, row in wp.iterrows():
    ax.annotate(str(row.get("name", "")),
                xy=(wp_x[i], wp_y[i]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

# Legend
terrain_legend = [
    mpatches.Patch(color="#d4b97a", label="Ground"),
    mpatches.Patch(color="#6db56d", label="Green/Natural"),
    mpatches.Patch(color="#999999", label="Road"),
    mpatches.Patch(color="#4a90d9", label="Water"),
    mpatches.Patch(color="#c0392b", label="Building"),
]
traj_legend = [
    plt.Line2D([0],[0], color="black", linewidth=1.5, label="Trajectory"),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="yellow",
               markeredgecolor="black", markersize=6, label="Waypoints"),
]
ax.legend(
    handles=terrain_legend + traj_legend,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    bbox_transform=ax.transAxes,   # ← locks to axes coords, immune to tight_layout
    borderaxespad=0,
    fontsize=8,
)

ax.set_xlabel("Easting (m, UTM 32N)")
ax.set_ylabel("Northing (m, UTM 32N)")
ax.set_title("Terrain map & drone trajectory")
plt.savefig("TERRAIN_MAP_with_trajectory.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# BLOCK 4 — Save trajectory for Phase 3
# =============================================================================

with open("trajectory_stage2.pkl", "wb") as f:
    pickle.dump({
        "traj_x":          traj_x,         # sampled x coords in UTM metres
        "traj_y":          traj_y,         # sampled y coords in UTM metres
        "traj_theta_deg": traj_theta_deg,
        "wp_x":            wp_x,           # waypoint x coords
        "wp_y":            wp_y,           # waypoint y coords
        "wp_names":        list(wp.get("name", pd.Series()).values),
        "sample_spacing":  SAMPLE_SPACING_M,
        "total_length_m":  total_length_m,
        "n_samples":       n_samples,
    }, f)
print("Saved trajectory_stage2.pkl")
