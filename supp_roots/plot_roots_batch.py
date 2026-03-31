import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def plot_roots(csv_name, output_pdf, title=None):
    filename = BASE_DIR / csv_name
    out_path = BASE_DIR / output_pdf

    fs = 20
    plt.rcParams.update({
        'font.size': fs,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold'
    })

    if not filename.exists():
        print(f"错误: 找不到文件 '{filename}'。请确保数据与本脚本同目录。")
        return

    data = pd.read_csv(filename, header=None, names=['l', 'Re', 'Im'])

    if data.empty:
        print("数据为空，没有可以绘制的根。")
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
        facecolors=(0.6, 0.6, 0.6, 0.4),
        edgecolors=(0, 0, 0, 0.4),
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

    if title is not None:
        plt.title(title, fontsize=fs)
    plt.xlabel(r'$\alpha R$', fontsize=fs)
    plt.ylabel(r'$\omega R$', fontsize=fs)
    plt.xlim(0, 3)
    plt.ylim(-10, 10)
    plt.axvline(0.5, linestyle='--', color='black', linewidth=1.0, alpha=0.8, zorder=1)
    plt.tight_layout()

    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"绘图完成！图片已保存为 '{out_path}'。")
    # plt.show()


if __name__ == '__main__':
    # plot_roots('roots_results000.csv', 'roots_complex_plane_mR_0p0.pdf',  r'$mR = 0.00$')
    # plot_roots('roots_results.csv',    'roots_complex_plane_mR_1p5.pdf',  r'$mR = 1.50$')
    # plot_roots('roots_results010.csv', 'roots_complex_plane_mR_0p10.pdf', r'$mR = 0.10$')
    # plot_roots('roots_results025.csv', 'roots_complex_plane_mR_0p25.pdf', r'$mR = 0.25$')
    plot_roots('roots_results035.csv', 'roots_complex_plane_mR_0p35.pdf', r'$mR = 0.35$')
    # plot_roots('roots_results05.csv',  'roots_complex_plane_mR_0p5.pdf',  r'$mR = 0.50$')
    # plot_roots('roots_results075.csv', 'roots_complex_plane_mR_0p75.pdf', r'$mR = 0.75$')
    # plot_roots('roots_results100.csv', 'roots_complex_plane_mR_1p0.pdf',  r'$mR = 1.00$')
    # plot_roots('roots_results125.csv', 'roots_complex_plane_mR_1p25.pdf', r'$mR = 1.25$')
    # plot_roots('roots_results175.csv', 'roots_complex_plane_mR_1p75.pdf', r'$mR = 1.75$')
    # plot_roots('roots_results200.csv', 'roots_complex_plane_mR_2p0.pdf',  r'$mR = 2.00$')




