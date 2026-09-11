########################################################################################################
############################################## Figure b-1 ##############################################
########################################################################################################
# %%
# plot_success_rate_barplot.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
from plot_figure_b_2 import ExperimentAnalyzer

def get_avg_and_stderr(analyzer, models, experiment_types):
    avg_list = []
    stderr_list = []
    for exp in experiment_types:
        vals = [analyzer.model_results.get(model, {}).get(exp, {}).get("success_rate", 0.0) / 100
                for model in models]
        avg = np.mean(vals)
        stderr = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        avg_list.append(avg)
        stderr_list.append(stderr)
    return avg_list, stderr_list

def plot_taskwise_comparison_barplot(
    dreamer_analyzer: ExperimentAnalyzer,
    ppo_analyzer: ExperimentAnalyzer,
    dqn_analyzer: ExperimentAnalyzer,
    human_analyzer: ExperimentAnalyzer,
    experiment_types: List[str],
    dreamer_models: List[str],
    ppo_models: List[str],
    dqn_models: List[str],
    save_path: Optional[str] = None):
    task_labels = [exp.replace("exp-", "") for exp in experiment_types]

    dreamer_avg = []
    for exp in experiment_types:
        vals = [dreamer_analyzer.model_results.get(model, {}).get(exp, {}).get("success_rate", 0.0)
                for model in dreamer_models]
        dreamer_avg.append(np.mean(vals))

    # Normalize all to [0, 1]
    dreamer_avg = [v / 100 for v in dreamer_avg]

    ppo_avg = []
    for exp in experiment_types:
        vals = [ppo_analyzer.model_results.get(model, {}).get(exp, {}).get("success_rate", 0.0)
                for model in ppo_models]
        ppo_avg.append(np.mean(vals))
    ppo_avg = [v / 100 for v in ppo_avg]

    dqn_avg = []
    for exp in experiment_types:
        vals = [dqn_analyzer.model_results.get(model, {}).get(exp, {}).get("success_rate", 0.0)
                for model in dqn_models]
        dqn_avg.append(np.mean(vals))
    dqn_avg = [v / 100 for v in dqn_avg]

    human_avg = []
    for exp in experiment_types:
        vals = [human_analyzer.model_results.get(model, {}).get(exp, {}).get("success_rate", 0.0)
                for model in human_list]
        human_avg.append(np.mean(vals))
    human_avg = [v / 100 for v in human_avg]

    x = np.arange(len(task_labels))
    width = 0.20
    loc = 0.20
    # width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    dreamer_avg, dreamer_stderr = get_avg_and_stderr(dreamer_analyzer, dreamer_models, experiment_types)
    ppo_avg, ppo_stderr = get_avg_and_stderr(ppo_analyzer, ppo_models, experiment_types)
    dqn_avg, dqn_stderr = get_avg_and_stderr(dqn_analyzer, dqn_models, experiment_types)
    human_avg, human_stderr = get_avg_and_stderr(human_analyzer, human_list, experiment_types)

    for i in range(len(x)):
        # DQN
        ax.bar(x[i] - loc, dqn_avg[i], width=width, color="#a2b8a4",
            label="DQN" if i == 0 else "", zorder=3,
            yerr=dqn_stderr[i], capsize=2,
            error_kw=dict(elinewidth=0.8, alpha=0.5) if len(dqn_models) <= 2 else dict(elinewidth=1))

        # PPO
        ax.bar(x[i], ppo_avg[i], width=width, color="#d7c9b1",
            label="PPO" if i == 0 else "", zorder=3,
            yerr=ppo_stderr[i], capsize=2,
            error_kw=dict(elinewidth=0.8, alpha=0.5) if len(ppo_models) <= 2 else dict(elinewidth=1))

        # Dreamer
        ax.bar(x[i] + loc, dreamer_avg[i], width=width, color="#5c6d84",
            label="Dreamer" if i == 0 else "", zorder=3,
            yerr=dreamer_stderr[i], capsize=2,
            error_kw=dict(elinewidth=1))

        # Human
        ax.bar(x[i] + loc*2, human_avg[i], width=width, color="#b3b3b3",
            label="Human" if i == 0 else "", zorder=3,
            yerr=human_stderr[i], capsize=2,
            error_kw=dict(elinewidth=1))

    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=30, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    based_dir = "Data_experiment"
    
    experiment_types = [
        'exp-two-resource-food', 'exp-two-resource-water', 'exp-two-resource-thermo',
        'exp-damage', 'exp-riskTaking', 'exp-goal-manipulation-FoodToWater',
        'exp-goal-manipulation-WaterToFood', 'exp-predator',
        'exp-Ymaze', 'exp-multiGoalPlanning',
    ]

    algo = "dreamer"
    dreamer_models = [
        "level-4-2", "level-4-1", "level-3-2", "level-3-1",
        "level-2-2", "level-2-1", "level-1-2", "level-1-1",
    ]
    dreamer_analyzer = ExperimentAnalyzer(os.path.join(based_dir, algo))
    dreamer_analyzer.get_model_dirs = lambda: dreamer_models
    dreamer_analyzer.analyze_all_models(experiment_types)

    algo = "ppo"
    ppo_models = ["level-1-1", "level-1-2"]
    ppo_analyzer = ExperimentAnalyzer(os.path.join(based_dir, algo))
    ppo_analyzer.get_model_dirs = lambda: ppo_models
    ppo_analyzer.analyze_all_models(experiment_types)

    algo = "dqn"
    dqn_models = ["level-1-1", "level-1-2"]
    dqn_analyzer = ExperimentAnalyzer(os.path.join(based_dir, algo))
    dqn_analyzer.get_model_dirs = lambda: dqn_models
    dqn_analyzer.analyze_all_models(experiment_types)


    human_base_dir = "human"
    human_list = ["sub1", "sub2", "sub3", "sub4", "sub5", "sub6", "sub7", "sub8"]

    human_analyzer = ExperimentAnalyzer(os.path.join(based_dir, human_base_dir))
    human_analyzer.get_model_dirs = lambda: human_list
    human_analyzer.analyze_all_models(experiment_types)

    plot_taskwise_comparison_barplot(
        dreamer_analyzer=dreamer_analyzer,
        ppo_analyzer=ppo_analyzer,
        dqn_analyzer=dqn_analyzer,
        experiment_types=experiment_types,
        dreamer_models=dreamer_models,
        ppo_models=ppo_models,
        dqn_models=dqn_models,
        human_analyzer=human_analyzer,
        save_path="figure_b_1.png"
    )
