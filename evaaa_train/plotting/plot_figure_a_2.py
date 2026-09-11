########################################################################################################
############################################## Figure a-2 ##############################################
########################################################################################################
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.patches import Rectangle

# Set base directory
seed_dir = "Data_training"
dpi = 300

# palette_ver1
dqn_color = "#a2b8a4"
ppo_color = "#d7c9b1"
dreamer_color = "#5c6d84"

# palette_ver2
curriculum_color ="#5c6d84" 
no_curriculum_color = "#b0b0b0"

Dreamer_configs = {
    "L1-1": {"color": dreamer_color, "path": "Dreamer_seed/level-1.1 scattered cubes no curriculum", "pattern": "dreamer*.csv"},
    "L1-2": {"color": dreamer_color, "path": "Dreamer_seed/level-1.2 corner-located cubes curriculum", "pattern": "dreamer*.csv"},
    "L2-1": {"color": dreamer_color, "path": "Dreamer_seed/level-2.1 natural obstacle layout curriculum", "pattern": "dreamer_v3*.csv"},
    "L2-2": {"color": dreamer_color, "path": "Dreamer_seed/level-2.2 obstacle-resource pairings curriculum", "pattern": "dreamer_v3*.csv"},
    "L3-1": {"color": dreamer_color, "path": "Dreamer_seed/level-3.1 navigation curriculum", "pattern": "dreamer*.csv"},
    "L3-2": {"color": dreamer_color, "path": "Dreamer_seed/level-3.2 exploration curriculum", "pattern": "dreamer*.csv"},
    "L4-1": {"color": dreamer_color, "path": "Dreamer_seed/level-4-1 predator curriculum", "pattern": "dreamer*.csv"},
    "L4-2": {"color": dreamer_color, "path": "Dreamer_seed/level-4-2 day and night curriculum", "pattern": "dreamer*.csv"},
    "L1-2(No)": {"color": dreamer_color, "path": "Dreamer_seed/level-1.2 corner-located cubes no curriculum", "pattern": "dreamer*.csv"},
    "L2-1(No)": {"color": dreamer_color, "path": "Dreamer_seed/level-1.2 corner-located cubes no curriculum", "pattern": "dreamer*.csv"},
    "L2-2(No)": {"color": dreamer_color, "path": "Dreamer_seed/level-2.2 obstacle-resource pairings no curriculum", "pattern": "dreamer_v3*.csv"},
    "L3-1(No)": {"color": dreamer_color, "path": "Dreamer_seed/level-3.1 navigation no curriculum", "pattern": "dreamer*.csv"},
    "L3-2(No)": {"color": dreamer_color, "path": "Dreamer_seed/level-3.2 exploration no curriculum", "pattern": "dreamer*.csv"},
    "L4-1(No)": {"color": dreamer_color, "path": "Dreamer_seed/level-4-1 predator no curriculum", "pattern": "dreamer*.csv"},
    "L4-2(No)": {"color": dreamer_color, "path": "Dreamer_seed/level-4-2 day and night no curriculum", "pattern": "dreamer*.csv"},
}


DQN_configs = {
    "L1-1": {"color": dqn_color, "path": "DQN_seed/dqn-level-1.1 scattered cubes no curriculum", "pattern": "DQN*.csv"},
    "L1-2": {"color": dqn_color, "path": "DQN_seed/dqn-level-1.2 corner-located cubes curriculum", "pattern": "DQN*.csv"},
    "L1-2(No)": {"color": dqn_color, "path": "DQN_seed/dqn-level-1.2 corner-located cubes no curriculum", "pattern": "DQN*.csv"},
}

PPO_configs = {
    "L1-1": {"color": ppo_color, "path": "PPO_seed/ppo-level-1.1 scattered cubes no curriculum", "pattern": "ppo*.csv"},
    "L1-2": {"color": ppo_color, "path": "PPO_seed/ppo-level-1.2 corner-located cubes curriculum", "pattern": "ppo*.csv"},
    "L1-2(No)": {"color": ppo_color, "path": "PPO_seed/ppo-level-1.2 corner-located cubes no curriculum", "pattern": "ppo*.csv"},
}

def load_top3_avg_and_se(base_dir, config_dict):
    avg_dict = {}
    se_dict = {}
    for label, config in config_dict.items():
        full_pattern = os.path.join(base_dir, config["path"], config["pattern"])
        files = glob.glob(full_pattern)

        run_means = []
        for f in files:
            try:
                df = pd.read_csv(f)
                top3 = df["Value"].nlargest(3).values
                run_means.append(np.mean(top3))
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if run_means:
            avg_dict[label] = np.mean(run_means)
            se_dict[label] = np.std(run_means, ddof=1) / np.sqrt(len(run_means))
        else:
            avg_dict[label] = 0
            se_dict[label] = 0
    return avg_dict, se_dict

DQN_avg, DQN_se = load_top3_avg_and_se(seed_dir, DQN_configs)
PPO_avg, PPO_se = load_top3_avg_and_se(seed_dir, PPO_configs)

DQN_labels = list(DQN_configs.keys())
PPO_labels = list(PPO_configs.keys())
DQN_colors = [DQN_configs[l]["color"] for l in DQN_labels]
PPO_colors = [PPO_configs[l]["color"] for l in PPO_labels]


Dreamer_avg = {}
Dreamer_se = {}
for label, config in Dreamer_configs.items():
    full_pattern = os.path.join(seed_dir, config["path"], config["pattern"])
    files = glob.glob(full_pattern)

    run_means = []
    for f in files:
        try:
            df = pd.read_csv(f)
            top3 = df["Value"].nlargest(3).values
            run_means.append(np.mean(top3))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if run_means:
        Dreamer_avg[label] = np.mean(run_means)
        Dreamer_se[label] = np.std(run_means, ddof=1) / np.sqrt(len(run_means))
    else:
        Dreamer_avg[label] = 0
        Dreamer_se[label] = 0

Dreamer_labels = list(Dreamer_configs.keys())

Dreamer_pretrain_labels = [l for l in Dreamer_labels if "(No)" not in l]
Dreamer_nopretrain_labels = [l for l in Dreamer_labels if "(No)" in l]

Dreamer_pretrain_colors = [Dreamer_configs[l]["color"] for l in Dreamer_pretrain_labels]
Dreamer_nopretrain_colors = [Dreamer_configs[l]["color"] for l in Dreamer_nopretrain_labels]

Dreamer_pretrain_avg = [Dreamer_avg[l] for l in Dreamer_pretrain_labels]
Dreamer_nopretrain_avg = [Dreamer_avg[l] for l in Dreamer_nopretrain_labels]

Dreamer_pretrain_se = [Dreamer_se[l] for l in Dreamer_pretrain_labels]
Dreamer_nopretrain_se = [Dreamer_se[l] for l in Dreamer_nopretrain_labels]

bar_width = 1.0
group_gap = 0.7
bar_spacing = 0.2

Dreamer_labels_plot = ["L1-1", "L1-2", "L2-1", "L2-2", "L3-1","L3-2","L4-1","L4-2","L1-2", "L2-1", "L2-2", "L3-1","L3-2","L4-1","L4-2"]

DQN_x = np.arange(len(DQN_labels)) * (bar_width + bar_spacing)
PPO_x = np.arange(len(PPO_labels)) * (bar_width + bar_spacing) + DQN_x[-1] + bar_width + group_gap
Dreamer_pretrain_x = np.arange(len(Dreamer_pretrain_labels)) * (bar_width + bar_spacing) + PPO_x[-1] + bar_width + group_gap
Dreamer_nopretrain_x = np.arange(len(Dreamer_nopretrain_labels)) * (bar_width + bar_spacing) + Dreamer_pretrain_x[-1] + bar_width + group_gap

all_x = np.concatenate([DQN_x, PPO_x, Dreamer_pretrain_x, Dreamer_nopretrain_x])
all_y = list(DQN_avg.values()) + list(PPO_avg.values()) + Dreamer_pretrain_avg + Dreamer_nopretrain_avg 
all_se = list(DQN_se.values()) + list(PPO_se.values()) + Dreamer_pretrain_se + Dreamer_nopretrain_se 
all_colors = DQN_colors + PPO_colors + Dreamer_pretrain_colors + Dreamer_nopretrain_colors 
all_labels = DQN_labels + PPO_labels + Dreamer_labels_plot 


# ------------------ Plotting ------------------ #
fig, ax = plt.subplots(figsize=(10,4), dpi=300)

ax.bar(all_x, all_y, bar_width, color=all_colors, yerr=all_se, capsize=5, alpha=0.9)

ax.set_ylim(0,70000)
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=12, rotation=45, ha='center')
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("figure_a_2.png", dpi=dpi)
plt.show()