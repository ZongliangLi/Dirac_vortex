"""Topological mode (infinite branch): read topo_inf.dat, compare to asymptotic theory."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_name = BASE_DIR / "topo_inf.dat"


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
    print(f"File not found: {file_name}. Place it next to this script.")
    raise SystemExit(1)

# Column 0: m/2 as x-axis m; column 1: Re(omega); column 2: Im(omega)
m = data[:, 0]
re_omega = data[:, 1]
im_omega = data[:, 2]

y_numeric = np.abs(im_omega)

# Asymptotic: 16 * m^2 * exp(-4m)
y_theory = 16 * (m**2) * np.exp(-4 * m)

fig, ax = plt.subplots(1, 1, figsize=(3.6, 3), dpi=300)
font_size = 10

m_min = float(np.min(m))
m_max = float(np.max(m))
span = m_max - m_min
m_dense = np.linspace(max(1e-8, m_min - 0.02 * span), m_max + 0.1 * span, 800)
y_theory_dense = 16 * (m_dense**2) * np.exp(-4 * m_dense)
ax.semilogy(
    m_dense,
    y_theory_dense,
    "-",
    color="#e74c3c",
    linewidth=1.8,
    zorder=2,
    label="Theory",
)

m_targets = np.linspace(1.5, 15.0, 20)
chosen_indices = []
for mt in m_targets:
    idx_closest = int(np.argmin(np.abs(m - mt)))
    chosen_indices.append(idx_closest)
idx_unique = np.unique(chosen_indices)

ax.semilogy(
    m[idx_unique],
    y_numeric[idx_unique],
    "o",
    color="#2980b9",
    markersize=4.0,
    markeredgewidth=0.0,
    zorder=3,
    label="Numerical",
)

ax.set_title("Zero mode", fontsize=font_size)
ax.set_xlabel(r"$mR$", fontsize=font_size)
ax.set_ylabel(r"$\alpha R$", fontsize=font_size)
ax.tick_params(axis="both", which="both", labelsize=font_size)
ax.grid(True, which="major", linestyle="-", alpha=0.18)
ax.legend(fontsize=font_size, frameon=False, loc="best")

plt.tight_layout(pad=0.2)
output_pdf = "zero_mode_imw.pdf"
plt.savefig(BASE_DIR / output_pdf, format="pdf", bbox_inches="tight")
plt.show()
