# =============================================================================
# PHASE 3 — Monte Carlo impact probability map
# =============================================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from physics.impact_point_ballistic import impact_point_ballistic_descent

# =============================================================================
# BLOCK 1 — Load Phase 1 and Phase 2 outputs
# =============================================================================

with open("terrain_stage1.pkl", "rb") as f:
    d1 = pickle.load(f)

with open("trajectory_stage2.pkl", "rb") as f:
    d2 = pickle.load(f)

x_min   = d1["x_min"];   x_max = d1["x_max"]
y_min   = d1["y_min"];   y_max = d1["y_max"]
terrain_grid = d1["terrain_grid"]

traj_x  = d2["traj_x"]   # (n_samples,) UTM metres
traj_y  = d2["traj_y"]   # (n_samples,) UTM metres
n_samples = d2["n_samples"]

traj_theta_deg = d2["traj_theta_deg"]   # (n_samples,) degrees

print(f"Trajectory points to simulate: {n_samples}")

# =============================================================================
# BLOCK 2 — Simulation parameters
# =============================================================================

# --- Output grid (independent of Phase 1 terrain grid) ---
N1 = 2000   # cells in x (easting)
N2 = 2000   # cells in y (northing)

x_limits = [x_min, x_max]
y_limits = [y_min, y_max]

# --- Physical parameters (payload) ---
g        = 9.81      # gravity (m/s^2)
densita  = 1.225     # air density (kg/m^3)
massa    = 1.0       # payload mass (kg)
superficie = 0.04    # frontal cross-sectional area of the payload (m^2)

# --- Uncertain parameters (mean, std) ---
altezza_mu    = 75.0;  altezza_sigma  = 5.0    # altitude at failure (m)
Cd_mu         = 1.0;   Cd_sigma       = 0.2    # drag coefficient
Vx_i_mu       = 17.0;  Vx_i_sigma    = 6.0   # horizontal velocity at separation (m/s)
Vy_i_mu       = 0.0;   Vy_i_sigma     = 0.5   # vertical velocity at separation (m/s)
mu_vento      = 15;   sigma_vento    = 0   # wind speed (m/s)
mu_drone      = 0;   sigma_drone    = 360 # wind direction (degrees)
# wind data taken from live updates by Osservatorio Metereologico UniTo: https://www.meteo.dfg.unito.it/completa


# --- Monte Carlo samples per trajectory point ---
GUESS = 5000   # Number of Monte Carlo simulations run at each trajectory point -> 1000+ for statistically meaningful results

# =============================================================================
# BLOCK 3 — Main loop: accumulate PDF over all trajectory points
# =============================================================================

global_pdf = np.zeros((N1, N2))

# Collectors for raw velocity and angle samples across all trajectory points
all_Vx     = []
all_Vy     = []
all_tim    = []



for i, (cx, cy) in enumerate(zip(traj_x, traj_y)):          # Iterates over every sampled trajectory point
    theta_i = traj_theta_deg[i]   # ← heading at this point

    pdf, _, _, _, Vx_raw, Vy_raw, tim_raw = impact_point_ballistic_descent(
        N1, N2,
        altezza_mu, altezza_sigma,
        massa, superficie,
        mu_vento, sigma_vento,
        mu_drone, sigma_drone,
        theta_i,               
        Cd_mu, Cd_sigma,
        Vx_i_mu, Vx_i_sigma,
        Vy_i_mu, Vy_i_sigma,
        cx, cy,
        GUESS, g, densita,
        x_limits, y_limits)

    global_pdf += pdf

    # Collect raw samples for PDF plots
    all_Vx.append(Vx_raw)
    all_Vy.append(Vy_raw)
    all_tim.append(tim_raw)
    if (i + 1) % 10 == 0 or (i + 1) == n_samples:
        print(f"  Simulated {i+1}/{n_samples} trajectory points...")

# Normalise so total probability sums to 1
total = global_pdf.sum()
if total > 0:
    global_pdf /= total

print(f"\nSimulation complete. Total probability mass: {global_pdf.sum():.4f}")

# Concatenate all raw samples into single arrays
all_Vx     = np.concatenate(all_Vx)       # (n_samples * GUESS,)
all_Vy     = np.concatenate(all_Vy)       # (n_samples * GUESS,)
all_tim    = np.concatenate(all_tim)

# =============================================================================
# BLOCK 4 — Visualise heatmap overlaid on terrain map
# =============================================================================

#terrain_cmap = mcolors.ListedColormap([
#    "#d4b97a", "#6db56d", "#999999", "#4a90d9", "#c0392b"
#])
terrain_cmap = mcolors.ListedColormap([
    "#e8c13a",   # 0 ground   — strong golden yellow
    "#1db833",   # 1 park     — vivid green
    "#505050",   # 2 road     — dark charcoal
    "#0055ff",   # 3 water    — strong bright blue
    "#ff1a1a",   # 4 building — vivid red
])
bounds_cmap = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm_terrain = mcolors.BoundaryNorm(bounds_cmap, terrain_cmap.N)

fig, ax = plt.subplots(figsize=(12, 9))

# Terrain base
ax.imshow(
    terrain_grid,
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap=terrain_cmap,
    norm=norm_terrain,
    interpolation="nearest",
    alpha=0.6)

# Impact probability heatmap
heatmap = ax.imshow(
    global_pdf.T * 100,          # transpose: histogram2d returns (x,y), imshow expects (row=y, col=x), IN PERCENTAGE
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap="hot_r",
    alpha=0.7,
    interpolation="bilinear")

plt.colorbar(heatmap, ax=ax, label="Impact probability (%)")

# Trajectory
ax.plot(traj_x, traj_y, color="cyan", linewidth=1.2, label="Trajectory", zorder=4)
ax.scatter(d2["wp_x"], d2["wp_y"],
           color="yellow", edgecolors="black", s=40, zorder=5, label="Waypoints")

# Legend
terrain_legend = [
    mpatches.Patch(color="#d4b97a", label="Ground"),
    mpatches.Patch(color="#6db56d", label="Green/Natural"),
    mpatches.Patch(color="#999999", label="Road"),
    mpatches.Patch(color="#4a90d9", label="Water"),
    mpatches.Patch(color="#c0392b", label="Building"),
]
traj_legend = [
    plt.Line2D([0],[0], color="cyan",  linewidth=1.5, label="Trajectory"),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="yellow",
               markeredgecolor="black", markersize=6, label="Waypoints"),
]
ax.legend(handles=terrain_legend + traj_legend, loc="upper left", fontsize=8)

ax.set_xlabel("Easting (m, UTM 32N)")
ax.set_ylabel("Northing (m, UTM 32N)")
ax.set_title(f"Impact probability map — {n_samples} trajectory points × {GUESS} simulations each")
plt.tight_layout()
plt.savefig("impact_heatmap.png", dpi=150)
plt.show()
print("Saved impact_heatmap.png")

# =============================================================================
# BLOCK 5 — Save outputs
# =============================================================================

with open("impact_stage3.pkl", "wb") as f:
    pickle.dump({
        "global_pdf": global_pdf,
        "N1": N1, "N2": N2,
        "x_limits": x_limits,
        "y_limits": y_limits,
    }, f)
print("Saved impact_stage3.pkl")

# =============================================================================
# BLOCK 6 — Impact velocity and time PDF plots
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Horizontal velocity
axes[0].hist(all_Vx, bins=80, density=True, color="steelblue", edgecolor="none")
axes[0].set_xlabel("Vx at impact (m/s)")
axes[0].set_ylabel("Probability density")
axes[0].set_title("Horizontal impact velocity")

# Vertical velocity (Vy is negative — downward)
axes[1].hist(all_Vy, bins=80, density=True, color="coral", edgecolor="none")
axes[1].set_xlabel("Vy at impact (m/s)")
axes[1].set_title("Vertical impact velocity")

# Impact time
axes[2].hist(all_tim, bins=80, density=True, color="seagreen", edgecolor="none")
axes[2].set_xlabel("Time of flight to impact (s)")
axes[2].set_title("Impact time")

plt.suptitle(f"Impact condition distributions — {n_samples} points × {GUESS} simulations",
             fontsize=11)
plt.tight_layout()
plt.savefig("chart_impact_velocity_pdf.png", dpi=150)
plt.show()
print("Saved chart_impact_velocity_pdf.png")


# =============================================================================
# BLOCK 7 — Impact distribution by terrain type
# =============================================================================

# Label encoding (must match Phase 1)
GROUND   = 0
PARK     = 1
ROAD     = 2
WATER    = 3
BUILDING = 4

terrain_names  = ["Ground", "Green/Natural", "Road", "Water", "Building"]
terrain_colors = ["#e8c13a", "#1db833", "#505050", "#0055ff", "#ff1a1a"]

# --- Load terrain grid cell size from Phase 1 ---
cell_w = d1["cell_w"]
cell_h = d1["cell_h"]
N_ROWS = d1["N_ROWS"]
N_COLS = d1["N_COLS"]

# --- For each cell in the impact heatmap, find which terrain type it belongs to ---
# The impact grid (N1 x N2) and terrain grid (N_ROWS x N_COLS) share the same
# bounding box but may have different resolutions. We map each impact grid cell
# to the nearest terrain grid cell.

# Impact grid cell centre coordinates
impact_xs = np.linspace(x_min, x_max, N1)
impact_ys = np.linspace(y_min, y_max, N2)

# Terrain grid cell centre coordinates
terrain_xs = d1["grid_x"][0, :]    # (N_COLS,) — x coords of terrain grid
terrain_ys = d1["grid_y"][:, 0]    # (N_ROWS,) — y coords of terrain grid

# Map each impact grid cell to nearest terrain grid cell
x_terrain_idx = np.searchsorted(terrain_xs, impact_xs) - 1
y_terrain_idx = np.searchsorted(terrain_ys, impact_ys) - 1

# Clip to valid range
x_terrain_idx = np.clip(x_terrain_idx, 0, N_COLS - 1)
y_terrain_idx = np.clip(y_terrain_idx, 0, N_ROWS - 1)

# Build terrain type lookup for every impact grid cell
# impact_terrain[i, j] = terrain label at impact cell (i, j)
impact_terrain = terrain_grid[
    np.ix_(y_terrain_idx, x_terrain_idx)
]   # shape (N2, N1) — matches global_pdf.T layout

# --- Compute probability mass per terrain type ---
# global_pdf is (N1, N2), impact_terrain is (N2, N1) so transpose one
pdf_by_terrain = np.zeros(5)

for label in range(5):
    mask = (impact_terrain == label).T   # (N1, N2) — matches global_pdf
    pdf_by_terrain[label] = global_pdf[mask].sum()

# Convert to percentage
pdf_by_terrain_pct = pdf_by_terrain * 100

print("\nImpact probability by terrain type:")
for label, name in enumerate(terrain_names):
    print(f"  {name:<10s}: {pdf_by_terrain_pct[label]:.2f}%")

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    terrain_names,
    pdf_by_terrain_pct,
    color=terrain_colors,
    edgecolor="black",
    linewidth=0.5
)

# Add value labels on top of each bar
for bar, val in zip(bars, pdf_by_terrain_pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{val:.1f}%",
        ha="center", va="bottom",
        fontsize=9
    )

ax.set_xlabel("Terrain type")
ax.set_ylabel("Cumulative impact probability (%)")
ax.set_title(f"Impact distribution by terrain type\n"
             f"{n_samples} trajectory points × {GUESS} simulations each")
ax.set_ylim(0, max(pdf_by_terrain_pct) * 1.15)
plt.tight_layout()
plt.savefig("chart_impact_by_terrain.png", dpi=150)
plt.show()
print("Saved chart_impact_by_terrain.png")