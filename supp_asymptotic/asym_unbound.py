"""Fit and plot unbound singlet: Im(omega) ~ A/m from data in asym_unbound.csv."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_cache_root = os.path.join(os.getcwd(), ".cache")
os.makedirs(_cache_root, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _cache_root)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache_root, "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "asym_unbound.csv"
if not file_path.exists():
    raise FileNotFoundError(f"Missing data file: {file_path}")

df = pd.read_csv(file_path)
m_vals = np.array(df["m"].values, dtype=np.float64)
im_vals = np.abs(np.array(df["Im_omega"].values, dtype=np.float64))

mask_fit = m_vals > 2e4
if not np.any(mask_fit):
    raise RuntimeError("No data with m > 2e4; cannot fit.")

m_fit = m_vals[mask_fit]
im_fit = im_vals[mask_fit]

n_I = 1.0
x_fit = 1.0 / m_fit
y_fit = im_fit
A_I = float(np.dot(x_fit, y_fit) / np.dot(x_fit, x_fit))

print("\n" + "=" * 60)
print("  Fit Im(omega) ~ A * m^(-1) with exponent fixed to -1")
print("=" * 60)
print(f"[1] Im(omega) ~ A * m^(-1)")
print(f"    Exponent n_I = {n_I:.6f}")
print(f"    A = {A_I:.5e}   (range m > 2e4)")
print("=" * 60 + "\n")

fig, ax = plt.subplots(1, 1, figsize=(3.6, 3), dpi=300)
font_size = 10

mR_vals = m_vals / 2.0

m_min = float(np.min(m_vals))
m_max = float(np.max(m_vals))
m_dense = np.logspace(np.log10(m_min / 1.05), np.log10(m_max * 1.6), 800)
mR_dense = m_dense / 2.0
theory_line_I = A_I * (m_dense ** (-n_I))
ax.loglog(
    mR_dense,
    theory_line_I,
    "-",
    color="#e74c3c",
    linewidth=1.8,
    zorder=2,
    label="Theory",
)

ks = np.linspace(np.log10(m_min), np.log10(m_max), 20)
m_targets = 10.0**ks
chosen_indices = []
for mt in m_targets:
    idx_closest = int(np.argmin(np.abs(m_vals - mt)))
    chosen_indices.append(idx_closest)
idx_unique = np.unique(chosen_indices)

ax.loglog(
    mR_vals[idx_unique],
    im_vals[idx_unique],
    "o",
    color="#2980b9",
    markersize=4.0,
    markeredgewidth=0.0,
    zorder=3,
    label="Numerical",
)

ax.set_xlabel(r"$mR$", fontsize=font_size)
ax.set_ylabel(r"$\alpha R$", fontsize=font_size)
ax.set_title("Unbound singlet", fontsize=font_size)
ax.tick_params(axis="both", which="both", labelsize=font_size)
ax.grid(True, which="major", linestyle="-", alpha=0.18)
ax.grid(False, which="minor")
ax.legend(fontsize=font_size, frameon=False, loc="best")

plt.tight_layout(pad=0.2)
output_pdf = "unbound_singlet_imw.pdf"
plt.savefig(BASE_DIR / output_pdf, format="pdf", bbox_inches="tight")
plt.show()
