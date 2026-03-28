import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import hankel2
from pathlib import Path

# 1. 载入原始数据
BASE_DIR = Path(__file__).resolve().parent
results_mR = np.load(BASE_DIR / 'results_mR.npy')
results_delta = np.load(BASE_DIR / 'results_delta.npy')

# 2. 你的新数据 (带反转)
new_mR = np.array([
    0.3000, 0.2927, 0.2864, 0.2811, 0.2765, 0.2726, 0.2693, 0.2665, 
    0.2640, 0.2620, 0.2602, 0.2587, 0.2574, 0.2563, 0.2554, 0.2546, 
    0.2539, 0.2534, 0.2529, 0.25
])[::-1]

new_delta = np.array([
    0.92008, 0.94056, 0.95661, 0.96874, 0.97763, 0.98400, 0.98853, 0.99176, 
    0.99407, 0.99574, 0.99694, 0.99781, 0.99843, 0.99888, 0.99921, 0.99944, 
    0.99960, 0.99972, 0.99981, 1
])[::-1]

# 3. 筛选并拼接数据
mask = results_mR > 0.3000
filtered_mR = results_mR[mask]
filtered_delta = results_delta[mask]

extended_mR = np.concatenate(([0.25], new_mR, filtered_mR))
extended_delta = np.concatenate(([1.0], new_delta, filtered_delta))




# ==========================================
# 重新设置字体：全局使用 Arial (包括公式)
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'Arial',        # 正文使用 Arial
    # -----------------------------------------------------------
    # 修改部分开始：将公式字体强制设为 Arial
    'mathtext.fontset': 'custom',  # 设置为自定义模式
    'mathtext.rm': 'Arial',        # 公式中的正体 (如 sin, cos, 单位) 使用 Arial
    'mathtext.it': 'Arial:italic', # 公式中的斜体变量 (如 x, mR, f1) 使用 Arial 斜体
    'mathtext.bf': 'Arial:bold',   # 公式中的粗体 使用 Arial 粗体
    # -----------------------------------------------------------
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'out',
    'ytick.direction': 'out'
})

c_orange = (240/256, 146/256, 53/256)
c_green = 'g'
c_red = 'r'

mR_array_61 = np.linspace(0, 6, 61) / 2
mR_array_550 = np.linspace(0.255, 3.0, 550)

# ------------------------------------------
# Load topo (zero mode) boundary loss data
# Combine points into mR_topo and topo_imag for mR in (0.5, 3]
# ------------------------------------------
def parse_value(val_str):
    """Parse Mathematica-exported rationals like '3/2' into float."""
    if '/' in val_str:
        num, den = val_str.split('/')
        return float(num) / float(den)
    return float(val_str)

def load_topo_im_from_scan(dat_path, imag_col_index=2):
    """
    Load (m, Im(omega)) from a whitespace-separated .dat file.
    Return (mR, Im(omega)).
    """
    data_list = []
    with open(dat_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            data_list.append([parse_value(p) for p in parts])
    data = np.array(data_list, dtype=float)
    m = data[:, 0]
    im_omega = data[:, imag_col_index]
    return m, im_omega

# Combine Omega_scan_results.dat and Omega_scan_towards_0.25_Adaptive.dat
# Merge their im_omega the same way as in asy_topo.py / asy_topo0.py:
# - asy_topo.py:     im_omega = data[:, 2]
# - asy_topo0.py:    im_omega = data[:, 2] * 2
TOPO_DIR = BASE_DIR.parent / "supp_asymptotic"
m_a, im_a = load_topo_im_from_scan(TOPO_DIR / 'topo_inf.dat', imag_col_index=2)  # Im(omega)
m_b, im_b = load_topo_im_from_scan(TOPO_DIR / 'topo_to0.dat', imag_col_index=2)  # Im(omega)

m_all = np.concatenate([m_a, m_b])
im_merged = np.concatenate([im_a, im_b * 2.0])

mask_topo = (m_all > 0.25) & (m_all <= 3.0)
mR_topo = m_all[mask_topo]
topo_imag = np.abs(im_merged[mask_topo])

# Sort and handle duplicate mR values by taking the smaller topo_imag (more stable branch)
sort_idx = np.argsort(mR_topo)
mR_sorted = mR_topo[sort_idx]
topo_sorted = topo_imag[sort_idx]

unique_mR, idx_start, counts = np.unique(mR_sorted, return_index=True, return_counts=True)
topo_min = np.minimum.reduceat(topo_sorted, idx_start)

mR_topo = unique_mR
topo_imag = topo_min

# ------------------------------------------
# Load omega branches from CSV (orange / green)
# CSV columns: [mR, Re(omega), Im(omega)]
# ------------------------------------------
omega_orange_csv = np.loadtxt(BASE_DIR / 'omega_orange.csv', delimiter=',')
omega_green_csv = np.loadtxt(BASE_DIR / 'omega_green.csv', delimiter=',')

mR_orange_csv = omega_orange_csv[:, 0]
omega_orange_imag_csv = omega_orange_csv[:, 2]*2

mR_green_csv = omega_green_csv[:, 0]
omega_green_real_csv = omega_green_csv[:, 1]
omega_green_imag_csv = omega_green_csv[:, 2]*2

# Interpolate onto the plotting grid mR_array_61
omega_orange_imag = np.interp(mR_array_61, mR_orange_csv, omega_orange_imag_csv)
omega_green_imag = np.interp(mR_array_61, mR_green_csv, omega_green_imag_csv)
omega_green_real = np.interp(mR_array_61, mR_green_csv, omega_green_real_csv)

# Interpolate topo_imag (zero mode) onto the plotting grid
topo_imag_on_grid = np.interp(mR_array_61, mR_topo, topo_imag, left=np.nan, right=np.nan)

# 下半部分差值计算
da_blue = omega_orange_imag - topo_imag_on_grid
da_green = omega_green_imag - topo_imag_on_grid
do = omega_green_real

# 修复：只屏蔽发散无效的极大值
invalid_mask = topo_imag_on_grid > 100
da_blue[invalid_mask] = np.nan
da_green[invalid_mask] = np.nan

# ==========================================
# 2. 创建双图布局 
# ==========================================
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 10.5), gridspec_kw={'height_ratios':[1, 1]})
plt.subplots_adjust(hspace=0.38) # 增加间距供插入图片

# 设置步长为 0.25 的 ticks
x_ticks = np.arange(0, 3.25, 0.25)
# 标签只显示 0.5 的倍数，避免互相拥挤重叠
x_labels =['0', '', '0.5', '', '1.0', '', '1.5', '', '2.0', '', '2.5', '', '3.0']

# ------------------------------------------
# 3. 绘制上方子图 Figure 3a (对数坐标)
# ------------------------------------------
ax_top_r = ax_top.twinx()  

# 绘制线条
ax_top_r.plot(extended_mR, extended_delta, color='k', linestyle='--')
ax_top.semilogy(mR_array_61, omega_orange_imag, color=c_orange)
ax_top.semilogy(mR_array_61, omega_green_imag, color=c_green)
ax_top.semilogy(mR_topo, topo_imag, color=c_red)

# 坐标轴设置
ax_top.set_xlim([0, 3])
ax_top.set_ylim([1e-3, 10])
ax_top_r.set_ylim([0, 1])

ax_top.set_xticks(x_ticks)
ax_top.set_xticklabels(x_labels)

# X轴标签在上方，刻度(ticks)在上下两边同时保留
ax_top.xaxis.set_label_position('top')
ax_top.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False, labeltop=True)

ax_top.set_xlabel(r'Real mass $mR$', labelpad=8)
ax_top.set_ylabel(r'Boundary loss $\alpha_{\parallel} R$', color='black')
# 现在的 f1 f2 将会以优美的斜体形式展现
ax_top_r.set_ylabel(r'Energy ratio $\bar{f_1^2}/\bar{f_2^2}$', color='black')

# ax_top_r.set_ylabel(r'Energy ratio $\bar{f_1^2}/\bar{f_2^2}$', color='black', math_fontfamily='stix')


ax_top.set_zorder(10)
ax_top_r.set_zorder(1)
ax_top.patch.set_visible(False)

# ---- 添加渐近关系文本标注 (Text Annotations) ----
fs = 14  # 调整为14与正文字体匹配更好

# 1/mR 发散标号 (红) 
ax_top.text(0.35, 4.9, r'$[8(mR-1/4)]^{-1}$', color=c_red, fontsize=fs, ha='left')

# 对齐的三个渐近线公式块 (同一 x 坐标左对齐)
align_x = 1.92
ax_top.text(align_x, 0.28, 'Unbound singlet\n' + r'$\sim (mR)^{-1}$', color=c_orange, fontsize=fs, ha='left')
ax_top.text(align_x, 0.032, 'Bound doublet\n' + r'$\sim (mR)^4 e^{-4/3mR}$', color=c_green, fontsize=fs, ha='left')
ax_top.text(align_x, 0.0012, 'Zero mode\n' + r'$ 16(mR)^2 e^{-4mR}$', color=c_red, fontsize=fs, ha='left')

ax_top.text(0.7, 0.012, r'$\bar{f_1^2}/\bar{f^2_2}$', color='k', fontsize=fs, ha='left')

# ------------------------------------------
# 4. 绘制下方子图 Figure 3b (线性坐标)
# ------------------------------------------
ax_bot_r = ax_bot.twinx() 

ax_bot.plot(mR_array_61[:32], da_blue[:32], '-', color=c_orange, zorder=2)
ax_bot.plot(mR_array_61[32:], da_blue[32:], '--', color=c_orange, zorder=2)
ax_bot.plot(mR_array_61[30:], da_green[30:], '-', color=c_green, zorder=2)
ax_bot.plot(mR_array_61[:30], da_green[:30], '--', color=c_green, zorder=2)
ax_bot_r.plot(mR_array_61, do, '-', color='k', zorder=1)

ax_bot.set_xlim([0, 3])
ax_bot.set_ylim([0, 0.3])
ax_bot_r.set_ylim([0, 3])

# 为了视觉上下绝对统一，下方图表也使用了 0.25 步长的 ticks
ax_bot.set_xticks(x_ticks)
ax_bot.set_xticklabels(x_labels)

ax_bot.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ax_bot_r.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

ax_bot.xaxis.set_label_position('bottom')
# 保留底图上方刻度
ax_bot.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False)

ax_bot.set_xlabel(r'Real mass $mR$')
ax_bot.set_ylabel(r'Threshold margin $\Delta \alpha R$', color='k')
ax_bot_r.set_ylabel(r'Free spectral range $\Delta \omega R$', color='k')

ax_bot.set_zorder(10)
ax_bot_r.set_zorder(1)
ax_bot.patch.set_visible(False)

# ---- 添加渐近关系文本标注 (Text Annotations) ----
ax_bot.text(2.5, 0.075, r'$\sim (mR)^4 e^{-4/3mR}$', color=c_green, fontsize=fs, ha='center')
ax_bot_r.text(2.65, 2.35, r'$\sim \sqrt{\frac{8}{9}} mR$', color='k', fontsize=16, ha='center')

# ==========================================
# 5. 最终对齐与输出
# ==========================================
fig.align_ylabels([ax_top, ax_bot])
fig.align_ylabels([ax_top_r, ax_bot_r])

# (a) (b) 标签也由于换了字体，会显得更正式
ax_top.text(-0.15, 1.05, '(a)', transform=ax_top.transAxes, fontsize=18, fontweight='normal', va='bottom')
ax_bot.text(-0.15, 1.05, '(b)', transform=ax_bot.transAxes, fontsize=18, fontweight='normal', va='bottom')

plt.savefig(BASE_DIR / 'boundary_loss_combined.pdf', bbox_inches='tight')
plt.show()
