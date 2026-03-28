"""Topological zero-mode divergence near mR = 1/4: read topo_to0.dat."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_name = BASE_DIR / "topo_to0.dat"


def parse_value(val_str):
    if "/" in val_str:
        num, den = val_str.split("/")
        return float(num) / float(den)
    return float(val_str)


data_list = []
try:
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            data_list.append([parse_value(p) for p in parts])
    data = np.array(data_list)
except FileNotFoundError:
    print(f"File not found: {file_name}.")
    raise SystemExit(1)

m = data[:, 0]
im_omega = data[:, 2] * 2

y_numeric = np.abs(im_omega)

m0 = 1.0 / 4.0
mask_range = (m >= m0) & (m <= 1.5) & (m > m0)
m_plot = m[mask_range]
y_plot = y_numeric[mask_range]

if m_plot.size == 0:
    raise RuntimeError("No data in (1/4, 1.5].")

delta_m = m_plot - m0
valid = delta_m > 0.0
delta_m = delta_m[valid]
y_plot = y_plot[valid]

if delta_m.size == 0:
    raise RuntimeError("No positive mR - 1/4 for log axis.")

y_theory = 1.0 / (8.0 * delta_m)

fig, ax = plt.subplots(1, 1, figsize=(3.6, 3), dpi=300)
font_size = 10

delta_min = float(np.min(delta_m))
delta_max = float(np.max(delta_m))
delta_left = max(delta_min / 1.05, 1e-12)
delta_right = delta_max * 1.6
delta_dense = np.logspace(np.log10(delta_left), np.log10(delta_right), 800)
y_theory_dense = 1.0 / (8.0 * delta_dense)
ax.loglog(
    delta_dense,
    y_theory_dense,
    "-",
    color="#e74c3c",
    linewidth=1.8,
    zorder=2,
    label="Theory",
)

delta_targets = np.logspace(np.log10(delta_min), np.log10(delta_max), 20)
chosen_indices = []
for dt in delta_targets:
    idx_closest = int(np.argmin(np.abs(delta_m - dt)))
    chosen_indices.append(idx_closest)
idx_unique = np.unique(chosen_indices)

ax.loglog(
    delta_m[idx_unique],
    y_plot[idx_unique],
    "o",
    color="#2980b9",
    markersize=4.0,
    markeredgewidth=0.0,
    zorder=3,
    label="Numerical",
)

ax.set_title("Zero mode", fontsize=font_size)
ax.set_xlabel(r"$mR - 1/4$", fontsize=font_size)
ax.set_ylabel(r"$\alpha R$", fontsize=font_size)
ax.tick_params(axis="both", which="both", labelsize=font_size)
ax.grid(True, which="major", linestyle="-", alpha=0.18)
ax.legend(fontsize=font_size, frameon=False, loc="best")

plt.tight_layout(pad=0.2)
output_pdf = "zero_mode_divergence_imw.pdf"
plt.savefig(BASE_DIR / output_pdf, format="pdf", bbox_inches="tight")
plt.show()
