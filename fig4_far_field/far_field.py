"""
Far-field patterns from the Whittaker radial solution: build psi(f1..f4, theta),
then radiation (E_l, E_r), Cartesian (radiation_x, radiation_y). The near field is
zero-padded (default 8x per side) before 2D FFT to refine k-space resolution, then
one figure is saved: intensity |far_x|^2+|far_y|^2 with polarization ellipses.

Convention for psi (same l, w, theta0 as in the notes):
  psi_1 = exp(i*(2l-w-1)/2*theta) * exp(-i*theta0/2) * f1
  psi_2 = exp(i*(2l-w+1)/2*theta) * exp(-i*theta0/2) * f2
  psi_3 = exp(i*(2l+w-1)/2*theta) * exp(+i*theta0/2) * f3
  psi_4 = exp(i*(2l+w+1)/2*theta) * (-exp(+i*theta0/2) * f4)

Radiation:
  E_l = psi_1*exp(-i*(theta+theta0)) - psi_3*exp(+i*(theta+theta0))
  E_r = -psi_2*exp(-i*(theta+theta0)) + psi_4*exp(+i*(theta+theta0))

  radiation_x = E_r + E_l
  radiation_y = 1j * E_r - 1j * E_l
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from whittaker_dirac_mm import (
    EN_INITIAL_GUESS,
    L,
    det_eqend,
    dirac_matrix_coeffs,
    f1234,
)

# --- Matplotlib cache (same pattern as whittaker_dirac_mm)
_cache = Path(__file__).resolve().parent / ".cache"
_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))

# ---------------------------------------------------------------------------
# User-tunable parameters (keep prominent)
# ---------------------------------------------------------------------------
# Global phase theta0 in the psi / radiation definitions (radians).
THETA0 = np.pi/6

# Integer parameter w in the azimuthal phase factors (fixed to 1 per problem statement).
W = 1

# Mass parameter m and boundary radius (same as whittaker_dirac_mm.main).
M_MASS = 17 / 10
R_MAX = 1.0
R_BC = 1.0

# Cartesian grid for near field: N x N samples on [-BOX_HALF, BOX_HALF]^2.
N_GRID = 128
BOX_HALF = 1.2

# Zero-pad the near field to PAD_FACTOR * N on each side (center embedding) before FFT.
# This refines k-space sampling (far-field resolution) without changing physical dx.
PAD_FACTOR = 8

# Avoid r=0 singularity when evaluating f_i(r).
R_INTERP_EPS = 1e-8

# Polarization ellipses: index stride on the padded FFT grid (smaller => denser inside the k-window).
POLA_STRIDE = 8  # was 16; halve stride => ~2x density along each axis
POLA_LENGTH = 0.2
POLA_LINEWIDTH = 1.0

# Far-field k-space window [-K_LIM, K_LIM] with K_LIM = pi/(2R) (R = cavity / normalization radius).
# Must match xlim/ylim; only draw pola where |k_x|,|k_y| <= K_LIM so ellipses are not off-screen.
def far_k_lim(radius: float) -> float:
    return float(np.pi / (2.0 * radius))


# Plot colors (visible on gray colormap)
POLA_COLOR = "darkred"
CMAP_INT = "gray"

# Single output basename (no separate Er-only / lr-only figures).
OUT_FAR_FIELD = "far_field_farfield"


def _solve_en_and_coeffs(
    m: float,
    ell: float,
    r_bc: float,
    en_guess: complex,
) -> tuple[complex, complex, complex]:
    """det=0 at r=r_bc; return (En, a1, a2) with a1=1."""

    def det_for_root(en_mpc):
        En = complex(en_mpc)
        try:
            d = det_eqend(r_bc, m, ell, En)
        except (ZeroDivisionError, ValueError, FloatingPointError):
            return mp.mpc(1e10, 1e10)
        return mp.mpc(d)

    sol = mp.findroot(det_for_root, en_guess, tol=1e-12, maxsteps=50, verify=False)
    en_val = complex(sol)
    mat = dirac_matrix_coeffs(r_bc, m, ell, en_val)
    a1 = 1.0 + 0j
    a2 = (-mat[0, 0] / mat[0, 1]) * a1
    return en_val, a1, a2


def psi_and_radiation_from_f(
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
    f4: np.ndarray,
    theta: np.ndarray,
    *,
    ell: float,
    w: float,
    theta0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build psi_1..psi_4 and (E_l, E_r) on the same grid as f_i, theta."""
    t = theta
    a = np.exp(1j * 0.5 * (2.0 * ell - w - 1.0) * t) * np.exp(-1j * 0.5 * theta0)
    b = np.exp(1j * 0.5 * (2.0 * ell - w + 1.0) * t) * np.exp(-1j * 0.5 * theta0)
    c = np.exp(1j * 0.5 * (2.0 * ell + w - 1.0) * t) * np.exp(1j * 0.5 * theta0)
    dph = np.exp(1j * 0.5 * (2.0 * ell + w + 1.0) * t)
    psi1 = a * f1
    psi2 = b * f2
    psi3 = c * f3
    psi4 = dph * (-np.exp(1j * 0.5 * theta0) * f4)

    ph_m = np.exp(-1j * (t + theta0))
    ph_p = np.exp(1j * (t + theta0))
    e_l = psi1 * ph_m - psi3 * ph_p
    e_r = -psi2 * ph_m + psi4 * ph_p
    return psi1, psi2, psi3, psi4, e_l, e_r


def build_near_field_cartesian(
    m: float,
    ell: float,
    en: complex,
    a1: complex,
    a2: complex,
    *,
    r_max: float,
    n: int,
    box_half: float,
    w: float,
    theta0: float,
    r_eps: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Sample f_i on a disk r<=r_max in the x-y plane; zero outside.
    Returns X, Y, radiation_x, radiation_y, E_l, E_r (complex 2D arrays).
    """
    x = np.linspace(-box_half, box_half, n, dtype=np.float64)
    y = np.linspace(-box_half, box_half, n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    r_eval = np.clip(r, r_eps, r_max)
    inside = r <= r_max

    f1, f2, f3, f4 = f1234(
        r_eval.astype(np.complex128),
        m,
        ell,
        en,
        a1,
        a2,
    )
    f1 = np.where(inside, f1, 0.0)
    f2 = np.where(inside, f2, 0.0)
    f3 = np.where(inside, f3, 0.0)
    f4 = np.where(inside, f4, 0.0)

    _p1, _p2, _p3, _p4, e_l, e_r = psi_and_radiation_from_f(
        f1, f2, f3, f4, theta, ell=float(ell), w=w, theta0=theta0
    )
    e_l = np.where(inside, e_l, 0.0)
    e_r = np.where(inside, e_r, 0.0)

    rad_x = e_r + e_l
    rad_y = 1j * e_r - 1j * e_l
    return X, Y, rad_x, rad_y, e_l, e_r


def far_field_fft2(z: np.ndarray) -> np.ndarray:
    out = np.fft.fft2(z)
    return np.fft.fftshift(out)


def pad_center_2d(a: np.ndarray, factor: int) -> np.ndarray:
    """Embed `a` in the center of a `factor` times larger square array (zero padding)."""
    n0, n1 = a.shape
    if n0 != n1:
        raise ValueError("pad_center_2d expects a square array")
    n = n0
    m = int(factor * n)
    pad = m - n
    p0 = pad // 2
    p1 = pad - p0
    return np.pad(a, ((p0, p1), (p0, p1)), mode="constant", constant_values=0.0)


def frequency_axes_1d(n: int, dx: float) -> np.ndarray:
    return np.fft.fftfreq(n, d=dx)


def pola(
    ex0: complex,
    ey0: complex,
    x0: float,
    y0: float,
    length: float,
    *,
    color: tuple[float, float, float] | str = POLA_COLOR,
    linewidth: float = 0.5,
    zorder: float = 5.0,
) -> None:
    """Draw a polarization ellipse (normalized) at (x0, y0)."""
    norm = np.sqrt(np.abs(ex0) ** 2 + np.abs(ey0) ** 2)
    if norm == 0.0 or not np.isfinite(norm):
        return
    ex = ex0 / norm * length
    ey = ey0 / norm * length
    ang = np.linspace(0.0, 2.0 * np.pi, 36)
    xs = np.real(ex * np.exp(-1j * ang)) + x0
    ys = np.real(ey * np.exp(-1j * ang)) + y0
    plt.plot(xs, ys, linewidth=linewidth, color=color, zorder=zorder)


def main() -> None:
    mp.mp.dps = 30

    ell = float(L)
    m = float(M_MASS)

    print("l =", ell, "| THETA0 =", THETA0, "| W =", W)
    print("findroot En initial guess =", EN_INITIAL_GUESS)

    en_val, a1_val, a2_val = _solve_en_and_coeffs(
        m, ell, R_BC, complex(EN_INITIAL_GUESS)
    )
    print("En =", en_val)
    print("a1 =", a1_val, ", a2 =", a2_val)

    x_1d = np.linspace(-BOX_HALF, BOX_HALF, N_GRID, dtype=np.float64)
    dx = float(x_1d[1] - x_1d[0])

    _X, _Y, rad_x, rad_y, _el, _er = build_near_field_cartesian(
        m,
        ell,
        en_val,
        a1_val,
        a2_val,
        r_max=R_MAX,
        n=N_GRID,
        box_half=BOX_HALF,
        w=float(W),
        theta0=float(THETA0),
        r_eps=R_INTERP_EPS,
    )

    rad_x_pad = pad_center_2d(rad_x, PAD_FACTOR)
    rad_y_pad = pad_center_2d(rad_y, PAD_FACTOR)
    m_pad = rad_x_pad.shape[0]

    far_x = far_field_fft2(rad_x_pad)
    far_y = far_field_fft2(rad_y_pad)
    intensity_xy = np.abs(far_x) ** 2 + np.abs(far_y) ** 2

    fx = frequency_axes_1d(m_pad, dx)
    fy = frequency_axes_1d(m_pad, dx)
    far_fx, far_fy = np.meshgrid(fx, fy, indexing="xy")
    far_fx_s = np.fft.fftshift(far_fx)
    far_fy_s = np.fft.fftshift(far_fy)

    out_dir = Path(__file__).resolve().parent

    k_lim = far_k_lim(R_MAX)
    print("Far-field k-window [-k_lim, k_lim] with k_lim = pi/(2R) =", k_lim)

    plt.figure(figsize=(5.0, 5.0))
    im = plt.pcolormesh(
        far_fx_s,
        far_fy_s,
        np.abs(intensity_xy),
        cmap=CMAP_INT,
        shading="auto",
        rasterized=True,
    )
    # Polarization grid: every POLA_STRIDE points, with index origin shifted by (n/2, n/2)
    # (left n/2 on kx / column j, up n/2 on ky / row i) to avoid sampling exactly at the DC corner.
    n = POLA_STRIDE
    i0 = n // 2
    j0 = n // 2
    for i in range(i0, m_pad, n):
        for j in range(j0, m_pad, n):
            kx = float(far_fx_s[i, j])
            ky = float(far_fy_s[i, j])
            if abs(kx) > k_lim or abs(ky) > k_lim:
                continue
            pola(
                far_x[i, j],
                far_y[i, j],
                kx,
                ky,
                POLA_LENGTH,
                linewidth=POLA_LINEWIDTH,
            )

    ax = plt.gca()
    # Square clip in k-space consistent with [-pi/(2R), pi/(2R)]
    clip_sq = patches.Rectangle(
        (-k_lim, -k_lim),
        2.0 * k_lim,
        2.0 * k_lim,
        transform=ax.transData,
        facecolor="none",
        edgecolor="none",
    )
    im.set_clip_path(clip_sq)
    for line in ax.lines:
        line.set_clip_path(clip_sq)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-k_lim, k_lim)
    ax.set_ylim(-k_lim, k_lim)
    ax.axis("off")
    plt.tight_layout(pad=0.0)

    png = out_dir / f"{OUT_FAR_FIELD}.png"
    plt.savefig(png, bbox_inches="tight", dpi=200)
    print("Saved", png)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
