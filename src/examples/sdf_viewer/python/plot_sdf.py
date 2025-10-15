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

# -------------------- Figure 4: Principal curvatures & directions -------------------
if all(k in data.dtype.names for k in ("hxx", "hxy", "hyx", "hyy")):
    # Use the already prepared HXX, HXY (symmetrized), HYY, plus X/Y extents
    A = HXX
    B = HXY
    C = HYY

    # Closed-form eigenvalues for symmetric 2x2 [[A,B],[B,C]]
    T  = A + C
    D  = np.sqrt((A - C)**2 + 4.0 * B**2)  # discriminant (>=0)
    Lmax = 0.5 * (T + D)
    Lmin = 0.5 * (T - D)

    # Corresponding eigenvectors (unnormalized). Choose a stable formula:
    # For lambda: v = [B, lambda - A]. If near-zero vector, fall back to [lambda - C, B].
    def eigvec_components(lmbd):
        vx = B
        vy = lmbd - A
        norm = np.hypot(vx, vy)
        # fallback where norm is tiny
        mask = norm < 1e-12
        vx_alt = lmbd - C
        vy_alt = B
        norm_alt = np.hypot(vx_alt, vy_alt)
        vx = np.where(mask, vx_alt, vx)
        vy = np.where(mask, vy_alt, vy)
        norm = np.where(mask, norm_alt, norm)
        vx /= np.where(norm == 0, 1.0, norm)
        vy /= np.where(norm == 0, 1.0, norm)
        return vx, vy

    Vx_max, Vy_max = eigvec_components(Lmax)
    Vx_min, Vy_min = eigvec_components(Lmin)

    # Plot
    QUIVER_EIG_STEP  = 12     # subsample for readability
    QUIVER_EIG_SCALE = 35.0   # arrow scaling
    ZOOM = (-0.7, 0.7, -0.7, 0.7)

    def robust_norm(Z):
        finite = Z[np.isfinite(Z)]
        if finite.size == 0:
            return 1.0
        lo, hi = np.percentile(finite, [1, 99])
        return max(abs(lo), abs(hi))

    # Build normalization
    zmax1 = robust_norm(Lmax); norm1 = TwoSlopeNorm(vmin=-zmax1, vcenter=0.0, vmax=zmax1)
    zmax2 = robust_norm(Lmin); norm2 = TwoSlopeNorm(vmin=-zmax2, vcenter=0.0, vmax=zmax2)

    # Create figure
    fig4, axes = plt.subplots(2, 1, figsize=(7.5, 9))
    plt.subplots_adjust(left=0.10, right=0.88, bottom=0.08, top=0.93, hspace=0.18)

    # Panel 1: Lmax + eigenvectors
    im1 = axes[0].imshow(
        Lmax, extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        origin="lower", cmap=COLORMAP, norm=norm1, interpolation="bilinear", aspect="equal"
    )
    axes[0].quiver(
        X[::QUIVER_EIG_STEP, ::QUIVER_EIG_STEP],
        Y[::QUIVER_EIG_STEP, ::QUIVER_EIG_STEP],
        Vx_max[::QUIVER_EIG_STEP, ::QUIVER_EIG_STEP],
        Vy_max[::QUIVER_EIG_STEP, ::QUIVER_EIG_STEP],
        pivot="mid", scale=QUIVER_EIG_SCALE, width=0.004
    )
    axes[0].set_title(r"Principal curvature $\lambda_{\max}$ with direction", fontsize=11)
    axes[0].set_xlim(ZOOM[0], ZOOM[1]); axes[0].set_ylim(ZOOM[2], ZOOM[3])

    # Panel 2: Lmin + eigenvectors
    im2 = axes[1].imshow(
        Lmin, extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        origin="lower", cmap=COLORMAP, norm=norm2, interpolation="bilinear", aspect="equal"
    )
    axes[1].quiver(
        X[::QUIVER_EIG_STEP, ::QUIVER_EIG_STEP],
        Y[::QUIVER_EIG_STEP, ::QUIVER_EIG_STEP],
        Vx_min[::QUIVER_EIG_STEP, ::QUIVER_EIG_STEP],
        Vy_min[::QUIVER_EIG_STEP, ::QUIVER_EIG_STEP],
        pivot="mid", scale=QUIVER_EIG_SCALE, width=0.004
    )
    axes[1].set_title(r"Principal curvature $\lambda_{\min}$ with direction", fontsize=11)
    axes[1].set_xlim(ZOOM[0], ZOOM[1]); axes[1].set_ylim(ZOOM[2], ZOOM[3])

    # Shared colorbars
    cbar_ax1 = fig4.add_axes([0.90, 0.56, 0.02, 0.32])
    cbar_ax2 = fig4.add_axes([0.90, 0.12, 0.02, 0.32])
    fig4.colorbar(im1, cax=cbar_ax1, label=r"$\lambda_{\max}$")
    fig4.colorbar(im2, cax=cbar_ax2, label=r"$\lambda_{\min}$")

    out_principal = (RESULTS_DIR / "hessian" / f"{STEM}_principal.png")
    fig4.suptitle(f"{CSV_FILENAME} — Principal curvatures & directions", fontsize=12)
    fig4.savefig(out_principal, dpi=300)
    print(f"Saved: {out_principal}")

plt.show()
