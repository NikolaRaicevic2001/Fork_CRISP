#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ------------------------------- CONFIG --------------------------------------
CSV_FILENAME    = "rounded_box_sdf_0p500_0p300_r_0p100.csv"  # file in results/
QUIVER_STEP     = 20                                         # subsample for quiver (larger = fewer arrows)
QUIVER_SCALE    = 40.0                                       # tweak arrow length

USE_COLORMAP    = True                                     
COLORMAP        = "coolwarm"                                # e.g., "RdBu_r", "seismic", "turbo", "viridis"
N_LEVELS        = 25                                        # number of levels (odd is nice to include 0)
# -----------------------------------------------------------------------------
THIS_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = THIS_DIR / "results"
CSV_PATH    = RESULTS_DIR / CSV_FILENAME
STEM        = Path(CSV_FILENAME).stem                  

# choose distinct save dirs & filenames based on mode
if USE_COLORMAP:
    OUT_DIR = RESULTS_DIR / "colormap"
    OUT_PNG = OUT_DIR / f"{STEM}_cmap.png"
else:
    OUT_DIR = RESULTS_DIR / "contour"
    OUT_PNG = OUT_DIR / f"{STEM}_contour.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
dmax = float(np.nanmax(np.abs(D))) or 1.0
norm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
levels = np.linspace(-dmax, dmax, N_LEVELS)

# Create figure and axis
fig, ax = plt.subplots()
if USE_COLORMAP:
    im = ax.contourf(X, Y, D, levels=levels, cmap=COLORMAP, norm=norm, antialiased=True)
    fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label="Signed distance")
else:
    cs = ax.contour(X, Y, D, levels=levels)
    ax.clabel(cs, inline=True, fontsize=8)

# Overlay the level-sets
cs = ax.contour(X, Y, D, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
ax.contour(X, Y, D, levels=[0.0], colors="k", linewidths=2.0)

# Optionally overlay normals
if QUIVER_STEP > 0:
    ax.quiver(
        X[::QUIVER_STEP, ::QUIVER_STEP],
        Y[::QUIVER_STEP, ::QUIVER_STEP],
        NX[::QUIVER_STEP, ::QUIVER_STEP],
        NY[::QUIVER_STEP, ::QUIVER_STEP],
        pivot="mid", scale=QUIVER_SCALE,
    )

ax.set_aspect("equal")
ax.set_title(CSV_FILENAME)
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)
plt.show()
print(f"Saved: {OUT_PNG}")
