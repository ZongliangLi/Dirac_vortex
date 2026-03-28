"""Fit and plot bound doublet: Im(omega) vs m from asymptotic data (asym_bound.csv)."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cache Matplotlib config under the project folder (avoids home-dir permission issues)
_cache_root = os.path.join(os.getcwd(), ".cache")
os.makedirs(_cache_root, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _cache_root)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache_root, "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "asym_bound.csv"
if not file_path.exists():
    raise FileNotFoundError(f"Missing data file: {file_path}")

df = pd.read_csv(file_path)
m_vals = df["m"].values
im_vals = np.abs(df["Im_omega"].values)
im_vals[im_vals == 0] = 1e-300

y_vals = im_vals * np.exp((2 / 3) * m_vals)

# Fix n = 4; fit only constant A on m > 50
mask = m_vals > 50
m_fit = m_vals[mask]
y_fit = y_vals[mask]

# y = A * m^4  =>  ln(y) = ln(A) + 4*ln(m)
ln_A = np.mean(np.log(y_fit) - 4 * np.log(m_fit))
A_fitted = np.exp(ln_A)

print("========== Fit with n = 4 fixed ==========")
print("Fit range: m > 50")
print(f"Coefficient A = {A_fitted:.5e}")
print(f"Approximation: y ~ {A_fitted:.3e} * m^4")
print("==========================================")

fig, ax = plt.subplots(1, 1, figsize=(3.6, 3), dpi=300)
font_size = 10

fit_im = A_fitted * (m_vals**4) * np.exp(-(2.0 / 3.0) * m_vals)

# Display: horizontal axis mR = m/2
mR_vals = m_vals / 2.0

m_min = float(np.min(m_vals))
m_max = float(np.max(m_vals))
span = m_max - m_min
m_dense = np.linspace(max(1e-8, m_min - 0.02 * span), m_max + 0.1 * span, 800)
mR_dense = m_dense / 2.0
fit_im_dense = A_fitted * (m_dense**4) * np.exp(-(2.0 / 3.0) * m_dense)

m_targets = np.linspace(m_min, m_max, 20)
chosen_indices = []
for mt in m_targets:
    idx_closest = int(np.argmin(np.abs(m_vals - mt)))
    chosen_indices.append(idx_closest)
idx_unique = np.unique(chosen_indices)

ax.semilogy(
    mR_dense,
    fit_im_dense,
    "-",
    color="#e74c3c",
    linewidth=1.8,
    zorder=2,
    label="Theory",
)
ax.semilogy(
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
ax.set_title("Bound doublet", fontsize=font_size)
ax.tick_params(axis="both", which="both", labelsize=font_size)
ax.grid(True, which="major", linestyle="-", alpha=0.18)
ax.legend(fontsize=font_size, frameon=False, loc="best")

plt.tight_layout(pad=0.2)
output_pdf = "bound_singlet_imw.pdf"
plt.savefig(BASE_DIR / output_pdf, format="pdf", bbox_inches="tight")
plt.show()
