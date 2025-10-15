#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ------------------------------- CONFIG --------------------------------------
CSV_FILENAME  = "circle_sdf_R0p350.csv" # file in results/
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

# ------------------- Figure 3: Hessian fields -------------------
if all(k in data.dtype.names for k in ("hxx", "hxy", "hyx", "hyy")):
    hxx = data["hxx"]; hxy = data["hxy"]; hyx = data["hyx"]; hyy = data["hyy"]

    HXX = hxx.reshape(H, W)
    HXY = hxy.reshape(H, W)
    HYX = hyx.reshape(H, W)
    HYY = hyy.reshape(H, W)
    HXY = 0.5 * (HXY + HYX)        # symmetrize
    LAP = HXX + HYY                # trace(H)

    DIR_HESS = RESULTS_DIR / "hessian"
    DIR_HESS.mkdir(parents=True, exist_ok=True)
    OUT_HESS_PNG = DIR_HESS / f"{STEM}_hessian.png"

    # --- plotting helper ---
    def draw_panel(ax, Z, title):
        finite_vals = Z[np.isfinite(Z)]
        if finite_vals.size == 0:
            zmax = 1.0
        else:
            lo, hi = np.percentile(finite_vals, [1, 99])  # robust clip
            zmax = max(abs(lo), abs(hi))
        normZ = TwoSlopeNorm(vmin=-zmax, vcenter=0.0, vmax=zmax)
        im = ax.imshow(Z, extent=(xs.min(), xs.max(), ys.min(), ys.max()),origin="lower", cmap=COLORMAP, norm=normZ,interpolation="bilinear", aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        return im

    # --- layout: tighter margins and larger center ---
    fig3, axs = plt.subplots(2, 2, figsize=(8, 7))
    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92, wspace=0.15, hspace=0.20)

    ims = [
        draw_panel(axs[0,0], HXX, r"$H_{xx}$"),
        draw_panel(axs[0,1], HXY, r"$H_{xy}$ (sym)"),
        draw_panel(axs[1,0], HYY, r"$H_{yy}$"),
        draw_panel(axs[1,1], LAP,  r"trace$(H)=H_{xx}+H_{yy}$")
    ]

    # share one colorbar for all
    cbar_ax = fig3.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig3.colorbar(ims[0], cax=cbar_ax, label="Hessian value")

    fig3.suptitle(f"{CSV_FILENAME} — Hessian fields", fontsize=12)
    fig3.savefig(OUT_HESS_PNG, dpi=300)
    print(f"Saved: {OUT_HESS_PNG}")
else:
    print("Hessian columns not found in CSV; skipping Hessian figure.")


plt.show()
