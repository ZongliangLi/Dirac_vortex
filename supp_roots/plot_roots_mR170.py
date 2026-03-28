"""
Plot roots for mR = 1.70 from roots_results170.csv.
Same styling as plot_roots_batch.py; axis limits: alpha R in [0, 0.5], omega R in [-3.5, 3.5].
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main():
    fs = 16
    plt.rcParams.update({
        'font.size': fs,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold'
    })

    filename = BASE_DIR / 'roots_results170.csv'
    output_pdf = BASE_DIR / 'roots_complex_plane_mR_1p7.pdf'

    if not filename.exists():
        print(f"错误: 找不到文件 '{filename}'。")
        return

    data = pd.read_csv(filename, header=None, names=['l', 'Re', 'Im'])
    if data.empty:
        print("数据为空。")
        return

    data['l'] = data['l'].astype(int)
    data['alpha'] = 2.0 * np.abs(data['Im'])
    data = data[data['alpha'] >= 0.01].copy()
    if data.empty:
        print("过滤 alpha R < 0.01 后无可用模式。")
        return

    mirrored = data.copy()
    mirrored['Re'] = -mirrored['Re']
    data_plot = pd.concat([data, mirrored], ignore_index=True)

    plt.figure(figsize=(6, 6), dpi=150)

    eps = 1e-10
    is_zero_mode = np.abs(data_plot['Re']) < eps
    is_l0 = data_plot['l'] == 0
    is_non_l0 = data_plot['l'] != 0

    subset_non_l0 = data_plot[is_non_l0 & (~is_zero_mode)]
    plt.scatter(
        subset_non_l0['alpha'],
        subset_non_l0['Re'],
        s=60,
        facecolors='0.6',
        edgecolors='black',
        linewidths=0.8,
        zorder=2
    )

    subset_l0 = data_plot[is_l0 & (~is_zero_mode)]
    plt.scatter(
        subset_l0['alpha'],
        subset_l0['Re'],
        s=60,
        facecolors='none',
        edgecolors='black',
        linewidths=1.5,
        zorder=3
    )

    subset_omega0 = data_plot[is_zero_mode]
    if not subset_omega0.empty:
        plt.scatter(
            subset_omega0['alpha'],
            subset_omega0['Re'],
            s=60,
            facecolors='none',
            edgecolors='red',
            linewidths=1.8,
            zorder=4
        )

    plt.xlabel(r'$\alpha R$', fontsize=14)
    plt.ylabel(r'$\omega R$', fontsize=14)
    plt.xlim(0, 0.5)
    plt.ylim(-3.5, 3.5)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"已保存: {output_pdf}")
    plt.show()


if __name__ == '__main__':
    main()
