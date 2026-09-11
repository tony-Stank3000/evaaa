########################################################################################################
############################################## Figure a-1 ##############################################
########################################################################################################
#%% 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# ========== Configurations ==========
seed_dir = "Data_training/Dreamer_seed"
max_episode = 700
smoothing_enabled = True
smoothing_window = 1
figure_size = (10, 4)
dpi = 300

neurips_palette = [
    "#d62728",  # 1-1
    "#1f77b4",  # 1-2
    "#8c564b",  # 2-1
    "#9467bd",  # 2-2
    "#bcbd22",  # 3-1
    "#2ca02c",  # 3-2
    "#ff7f0e",  # 4-1
    "#17becf",  # 4-2
]

Dreamer_configs = {
    "L1-1: Scattered Cubes": {"line_style": "-", "color": neurips_palette[0], "alpha": 0.9,
        "path": "level-1.1 scattered cubes no curriculum", "pattern": "dreamer*.csv"},
    "L1-2: Edge Cubes": {"line_style": "-", "color": neurips_palette[1], "alpha": 0.9,
        "path": "level-1.2 corner-located cubes curriculum", "pattern": "dreamer*.csv"},
    "L2-1: Obstacles": {"line_style": "-", "color": neurips_palette[2], "alpha": 0.9,
        "path": "level-2.1 Obstacles curriculum", "pattern": "dreamer*.csv"},
    "L2-2: Resource Mapping": {"line_style": "-", "color": neurips_palette[3], "alpha": 0.9,
        "path": "level-2.2 resource mapping curriculum", "pattern": "dreamer*.csv"},
    "L3-1: Navigation": {"line_style": "-", "color": neurips_palette[4], "alpha": 0.7,
        "path": "level-3.1 navigation curriculum", "pattern": "dreamer*.csv"},
    "L3-2: Exploration": {"line_style": "-", "color": neurips_palette[5], "alpha": 0.7,
        "path": "level-3.2 exploration curriculum", "pattern": "dreamer*.csv"},
    "L4-1: Predator": {"line_style": "-", "color": neurips_palette[6], "alpha": 0.7,
        "path": "level-4-1 predator curriculum", "pattern": "dreamer*.csv"},
    "L4-2: Day and Night": {"line_style": "-", "color": neurips_palette[7], "alpha": 0.7,
        "path": "level-4-2 day and night curriculum", "pattern": "dreamer*.csv"},

}

def smooth_curve(y, window_size):
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')

def padded_mean_std(value_list):
    max_len = max(len(v) for v in value_list)
    padded = np.full((len(value_list), max_len), np.nan)
    for i, v in enumerate(value_list):
        padded[i, :len(v)] = v
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    return mean, std

def process_seed_runs(file_paths):
    all_values = []
    all_steps = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        values = df["Value"].values
        steps = df["Step"].values
        if smoothing_enabled and len(values) > smoothing_window:
            values = smooth_curve(values, smoothing_window)
            steps = steps[:len(values)]
        mask = steps <= max_episode
        values = values[:np.sum(mask)]
        steps = steps[:np.sum(mask)]
        all_values.append(values)
        all_steps.append(steps)

    ref_steps = all_steps[np.argmax([len(s) for s in all_steps])]
    mean, std = padded_mean_std(all_values)
    return ref_steps, mean, std

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=figure_size, dpi=dpi,
    gridspec_kw={'width_ratios': [300, 50]}  
)

for label, cfg in Dreamer_configs.items():
    files = sorted(glob.glob(os.path.join(seed_dir, cfg["path"], cfg["pattern"])))
    if not files:
        continue
    steps, mean_values, std_values = process_seed_runs(files)

    mask1 = steps <= 300
    ax1.plot(steps[mask1], mean_values[mask1], color=cfg["color"], linestyle=cfg["line_style"], alpha=cfg["alpha"], label=label)
    ribbon_scale = 0.3
    ax1.fill_between(steps[mask1],
                    mean_values[mask1] - ribbon_scale * std_values[mask1],
                    mean_values[mask1] + ribbon_scale * std_values[mask1],
                    color=cfg["color"], alpha=0.1)
    
    mask2 = steps >= 575
    ax2.plot(steps[mask2], mean_values[mask2], color=cfg["color"], linestyle=cfg["line_style"], alpha=cfg["alpha"])
    ax2.fill_between(steps[mask2],
                mean_values[mask2] - ribbon_scale * std_values[mask2],
                mean_values[mask2] + ribbon_scale * std_values[mask2],
                color=cfg["color"], alpha=0.1)

# Broken axis diagonal lines
for ax in [ax1, ax2]:
    ax.spines['right' if ax == ax1 else 'left'].set_visible(False)
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Axis limits and labels
ax1.set_xlim(0, 300)
ax2.set_xlim(575, max_episode)
ax1.set_ylim(0, 50000)
for ax in (ax1, ax2):
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
# ax.grid(axis='y', linestyle='--', alpha=0.3)

ax2.legend(loc='upper right', fontsize=11, frameon=False)
plt.tight_layout()
plt.savefig("figure_a_1.png", dpi=dpi)
plt.show()
