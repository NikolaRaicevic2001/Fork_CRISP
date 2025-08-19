#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ------------------------------- CONFIG --------------------------------------
CSV_FILENAME  = "roundedsmooth_sdf_0p050_0p050_r_0p010.csv" # file in results/
QUIVER_STEP   = 20
QUIVER_SCALE  = 40.0
COLORMAP      = "coolwarm"                                  # e.g., "RdBu_r", "seismic", "turbo", "viridis"
N_LEVELS      = 25
# -----------------------------------------------------------------------------

# Paths
THIS_DIR    = Path(__file__).resolve().parent.parent  
RESULTS_DIR = THIS_DIR / "results"
CSV_PATH    = RESULTS_DIR / CSV_FILENAME
STEM        = Path(CSV_FILENAME).stem

DIR_CMAP    = RESULTS_DIR / "colormap"
DIR_CONTOUR = RESULTS_DIR / "contour"
DIR_CMAP.mkdir(parents=True, exist_ok=True)
DIR_CONTOUR.mkdir(parents=True, exist_ok=True)

OUT_CMAP_PNG    = DIR_CMAP    / f"{STEM}_cmap.png"
OUT_CONTOUR_PNG = DIR_CONTOUR / f"{STEM}_contour.png"

# Load CSV (expects header: x,y,d,nx,ny)
data = np.genfromtxt(CSV_PATH, delimiter=",", names=True)
x, y, d, nx, ny = data["x"], data["y"], data["d"], data["nx"], data["ny"]

# Build grid
xs = np.sort(np.unique(x))
ys = np.sort(np.unique(y))
W, H = len(xs), len(ys)
X  = xs[None, :].repeat(H, axis=0)
Y  = ys[:, None].repeat(W, axis=1)
D  = d.reshape(H, W)
NX = nx.reshape(H, W)
NY = ny.reshape(H, W)

# Symmetric normalization centered at 0 so colors flip sign nicely
dmax   = float(np.nanmax(np.abs(D))) or 1.0
norm   = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
levels = np.linspace(-dmax, dmax, N_LEVELS)

# ------------------- Figure 1: Filled colormap -------------------
fig1, ax1 = plt.subplots()
im = ax1.contourf(X, Y, D, levels=levels, cmap=COLORMAP, norm=norm, antialiased=True)
fig1.colorbar(im, ax=ax1, shrink=0.9, pad=0.02, label="Signed distance")
# overlay zero level & light level sets
ax1.contour(X, Y, D, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
ax1.contour(X, Y, D, levels=[0.0], colors="k", linewidths=2.0)
# normals
if QUIVER_STEP > 0:
    ax1.quiver(
        X[::QUIVER_STEP, ::QUIVER_STEP],
        Y[::QUIVER_STEP, ::QUIVER_STEP],
        NX[::QUIVER_STEP, ::QUIVER_STEP],
        NY[::QUIVER_STEP, ::QUIVER_STEP],
        pivot="mid", scale=QUIVER_SCALE,
    )
ax1.set_aspect("equal")
ax1.set_title(f"{CSV_FILENAME} — colormap")
fig1.tight_layout()
fig1.savefig(OUT_CMAP_PNG, dpi=300)

# ------------------- Figure 2: Line contours -------------------
fig2, ax2 = plt.subplots()
cs = ax2.contour(X, Y, D, levels=levels)
ax2.clabel(cs, inline=True, fontsize=8)
ax2.contour(X, Y, D, levels=[0.0], colors="k", linewidths=2.0)
if QUIVER_STEP > 0:
    ax2.quiver(
        X[::QUIVER_STEP, ::QUIVER_STEP],
        Y[::QUIVER_STEP, ::QUIVER_STEP],
        NX[::QUIVER_STEP, ::QUIVER_STEP],
        NY[::QUIVER_STEP, ::QUIVER_STEP],
        pivot="mid", scale=QUIVER_SCALE,
    )
ax2.set_aspect("equal")
ax2.set_title(f"{CSV_FILENAME} — contours")
fig2.tight_layout()
fig2.savefig(OUT_CONTOUR_PNG, dpi=300)

print(f"Saved: {OUT_CMAP_PNG}")
print(f"Saved: {OUT_CONTOUR_PNG}")

plt.show()
