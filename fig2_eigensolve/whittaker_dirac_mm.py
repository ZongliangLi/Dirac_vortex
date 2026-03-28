"""
Python translation of the provided Mathematica notebook:

  z = 4 sqrt(m^2 - En^2) r
  alpha = m / (2 sqrt(m^2 - En^2))
  y1 = r^(-1/2) M[alpha,l,z] a1,  y2 = r^(-1/2) M[-alpha,l,z] a2
  f2 = y1 + y2,  f3 = -I (y1 - y2)

  f1 and f4 are implemented via an equivalent Whittaker closed form (_f1_f4_closed),
  avoiding finite-difference error from differentiating f2 and f3.

Boundary matrix from coefficients of (f1+f2) and (f3+f4) in (a1,a2); det = 0
for En; then a2/a1 from the first row; plots f1..f4 vs r.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

_cache = Path(__file__).resolve().parent / ".cache"
_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))

# Initial guess for mpmath.findroot: det(boundary) = 0 at r = r_bc in main().
# Edit this (and/or mpmath mp.dps) to converge to a different eigenvalue En.
# Examples: 1j, 0.02j, 0.05 + 0.02 * 1j
EN_INITIAL_GUESS = 0+0.1j

# Angular momentum: second index l in Whittaker M_{kappa, l}(z) (same l as in the radial ansatz).
L = 0

# If True, also save Re/Im plots of f1..f4 (whittaker_dirac_mm_f1234.pdf). Default: intensity only.
PLOT_F_COMPONENTS = False


def whittaker_M(kappa, mu, z):
    """Whittaker M_{kappa,mu}(z); z can be an array."""
    z = np.asarray(z, dtype=np.complex128)
    flat = z.ravel()
    out = np.empty(flat.shape, dtype=np.complex128)
    kappa_m = mp.mpc(kappa)
    mu_m = mp.mpc(mu)
    for i, zi in enumerate(flat):
        out[i] = complex(mp.whitm(kappa_m, mu_m, mp.mpc(zi)))
    return out.reshape(z.shape)


def _f2f3(r, m, l, En, a1, a2):
    r = np.asarray(r, dtype=np.complex128)
    En = complex(En)
    m = float(m)
    l = complex(l)
    a1 = complex(a1)
    a2 = complex(a2)
    S = np.sqrt(m**2 - En**2)
    nu = m / (2.0 * S)
    z = 4.0 * r * S
    mp_ = whittaker_M(nu, l, z)
    mn_ = whittaker_M(-nu, l, z)
    inv_sqrt_r = r ** (-0.5)
    y1 = inv_sqrt_r * mp_ * a1
    y2 = inv_sqrt_r * mn_ * a2
    f2 = y1 + y2
    f3 = -1j * (y1 - y2)
    return f2, f3


def _f1_f4_closed(r_arr, m, l, En, a1, a2):
    """
    Closed-form Whittaker expression for f1 and f4 (algebraically equivalent to
    the Dirac relations differentiated in f2, f3). Uses S = sqrt(m^2-E^2),
    nu = m/(2S), z = 4 r S, A1 = a1, A2 = a2.
    """
    r_arr = np.asarray(r_arr, dtype=np.complex128)
    m = float(m)
    l = complex(l)
    En = complex(En)
    a1 = complex(a1)
    a2 = complex(a2)
    S = np.sqrt(m**2 - En**2)
    nu = m / (2.0 * S)
    z = 4.0 * r_arr * S

    M_nu = whittaker_M(nu, l, z)
    M_1pnu = whittaker_M(1.0 + nu, l, z)
    M_mnu = whittaker_M(-nu, l, z)
    M_1mnu = whittaker_M(1.0 - nu, l, z)

    C11 = (
        -4.0 * m**2 * r_arr
        + 4.0 * r_arr * En**2
        + (1.0 - 2.0 * l) * S
        + m * (1.0 - 4.0 * r_arr * S)
    )
    C12 = m + (1.0 + 2.0 * l) * S
    C13 = (
        -m
        - 4.0 * m**2 * r_arr
        + 4.0 * r_arr * En**2
        + 4.0 * m * r_arr * S
        + (1.0 - 2.0 * l) * S
    )
    C14 = m - (1.0 + 2.0 * l) * S

    C41 = (
        -4.0 * m**2 * r_arr
        + 4.0 * r_arr * En**2
        + (1.0 + 2.0 * l) * S
        + m * (1.0 - 4.0 * r_arr * S)
    )
    C42 = C12
    C43 = (
        -m
        - 4.0 * m**2 * r_arr
        + 4.0 * r_arr * En**2
        + 4.0 * m * r_arr * S
        + (1.0 + 2.0 * l) * S
    )
    C44 = C14

    pref = 1.0 / (4.0 * r_arr**1.5 * En * S)
    bracket1 = a1 * (C11 * M_nu - C12 * M_1pnu) + a2 * (C13 * M_mnu + C14 * M_1mnu)
    bracket4 = a1 * (C41 * M_nu - C42 * M_1pnu) - a2 * (C43 * M_mnu + C44 * M_1mnu)
    f1 = 1j * pref * bracket1
    f4 = pref * bracket4
    return f1, f4


def f1234(r, m, l, En, a1, a2):
    """Return f1..f4 for a scalar or 1D array r; f1 and f4 use the closed form above."""
    r_in = np.asarray(r, dtype=np.complex128)
    was_scalar = r_in.ndim == 0
    r_arr = np.atleast_1d(r_in)

    f2, f3 = _f2f3(r_arr, m, l, En, a1, a2)
    f1, f4 = _f1_f4_closed(r_arr, m, l, En, a1, a2)

    if was_scalar:
        return f1[0], f2[0], f3[0], f4[0]
    return f1, f2, f3, f4


def dirac_matrix_coeffs(r, m, l, En):
    """Boundary matrix: column j = (a1,a2) = e_j."""
    f1, f2, f3, f4 = f1234(r, m, l, En, 1.0, 0.0)
    d11, d21 = f1 + f2, f3 + f4
    f1, f2, f3, f4 = f1234(r, m, l, En, 0.0, 1.0)
    d12, d22 = f1 + f2, f3 + f4
    return np.array([[d11, d12], [d21, d22]], dtype=np.complex128)


def det_eqend(r, m, l, En):
    M = dirac_matrix_coeffs(r, m, l, En)
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def plot_sum_sq_mod_f1234(
    r_pos,
    f1,
    f2,
    f3,
    f4,
    *,
    R: float,
    symmetric: bool = True,
    xlim: tuple[float, float] | None = None,
    color: str = "red",
    color2: str = "red",
    spine_color: str = "red",
    figsize: tuple[float, float] = (6.2, 4.0),
    outfile: Path | None = None,
) -> plt.Figure:
    """
    Plot sum_i |f_i(r)|^2. r_pos samples the physical radius on (0, R].

    If symmetric is True, draw x in [-R, R] with I(-x) = I(|x|) (mirror of [0, R]).
    If False, plot vs r_pos only; set xlim or it defaults to (0, R).

    Curve and fill use red by default; left/right spines use spine_color (default red).
    """
    f_intensity = (
        np.abs(np.asarray(f1, dtype=np.complex128)) ** 2
        + np.abs(np.asarray(f2, dtype=np.complex128)) ** 2
        + np.abs(np.asarray(f3, dtype=np.complex128)) ** 2
        + np.abs(np.asarray(f4, dtype=np.complex128)) ** 2
    )
    r_pos = np.asarray(r_pos, dtype=float)

    if symmetric:
        x_all = np.concatenate([-r_pos[::-1], r_pos])
        f_plot = np.concatenate([f_intensity[::-1], f_intensity])
        xlim_use = (-float(R), float(R))
    else:
        x_all = r_pos
        f_plot = f_intensity
        xlim_use = xlim if xlim is not None else (0.0, float(R))

    ymax = float(np.max(f_plot)) if f_plot.size else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_all, f_plot, color=color, linewidth=8)
    ax.set_xlim(xlim_use)
    ax.set_ylim((0.0, ymax if ymax > 0.0 else 1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.fill_between(x_all, f_plot, color=color2, alpha=0.2)
    ax.spines["left"].set_color(spine_color)
    ax.spines["right"].set_color(spine_color)
    ax.spines["left"].set_linewidth(8)
    ax.spines["right"].set_linewidth(8)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")
        print("Saved", outfile)
    return fig


def main():
    mp.mp.dps = 30

    m = 17 / 10
    l_val = L
    r_bc = 1.0

    def det_for_root(en_mpc):
        En = complex(en_mpc)
        try:
            d = det_eqend(r_bc, m, l_val, En)
        except (ZeroDivisionError, ValueError, FloatingPointError):
            return mp.mpc(1e10, 1e10)
        return mp.mpc(d)

    print("l =", l_val, "| findroot(det=0), En initial guess =", EN_INITIAL_GUESS)
    # Default verify step is very strict; relax tol and skip verify when residual ~1e-14.
    sol = mp.findroot(
        det_for_root, EN_INITIAL_GUESS, tol=1e-12, maxsteps=50, verify=False
    )
    En_val = complex(sol)
    print("En =", En_val)

    mat = dirac_matrix_coeffs(r_bc, m, l_val, En_val)
    a1_val = 1.0 + 0j
    a2_val = (-mat[0, 0] / mat[0, 1]) * a1_val
    print("a1 = 1, a2 =", a2_val)

    R = 1.0
    r_eps = 1e-8
    r_plot = np.linspace(r_eps, R, 500)
    f1p, f2p, f3p, f4p = f1234(r_plot, m, l_val, En_val, a1_val, a2_val)

    if PLOT_F_COMPONENTS:
        labels = [r"$f_1$", r"$f_2$", r"$f_3$", r"$f_4$"]
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        series = [
            (f1p, labels[0], colors[0]),
            (f2p, labels[1], colors[1]),
            (f3p, labels[2], colors[2]),
            (f4p, labels[3], colors[3]),
        ]

        fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True)
        for fk, lab, c in series:
            ax_re.plot(r_plot, np.real(fk), color=c, label=lab, linewidth=1.3)
            ax_im.plot(r_plot, np.imag(fk), color=c, linewidth=1.3)
        ax_re.set_ylabel(r"$\mathrm{Re}(f_i)$")
        ax_im.set_ylabel(r"$\mathrm{Im}(f_i)$")
        ax_im.set_xlabel(r"$r$")
        ax_re.legend(loc="best", fontsize=9, ncol=2, frameon=False)
        ax_re.grid(True, linestyle="--", alpha=0.35)
        ax_im.grid(True, linestyle="--", alpha=0.35)
        ax_re.set_xlim(0.0, R)
        fig.tight_layout()
        out = Path(__file__).resolve().parent / "whittaker_dirac_mm_f1234.pdf"
        plt.savefig(out, bbox_inches="tight")
        print("Saved", out)

    out_i = Path(__file__).resolve().parent / "whittaker_dirac_mm_f1234_intensity.pdf"
    plot_sum_sq_mod_f1234(r_plot, f1p, f2p, f3p, f4p, R=R, symmetric=True, outfile=out_i)

    plt.show()


if __name__ == "__main__":
    main()
