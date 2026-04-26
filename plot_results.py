import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd

def ewa(arr: np.ndarray, weight: float) -> np.ndarray:
    arr_ewa = np.empty_like(arr)
    arr_ewa[0] = arr[0]
    for t in range(1, len(arr)):
        arr_ewa[t] = weight * arr[t] + (1 - weight) * arr_ewa[t - 1]
    return arr_ewa


mpl.use("pgf")
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "font.size": 14,
    }
)
stage_1_results = pd.read_csv("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_SP/PPO_Soccer_e492d_00000_0_2026-03-17_22-40-57/progress.csv")["episode_reward_mean"]
stage_2_results = pd.read_csv("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_rec/PPO_Soccer_bd183_00000_0_2026-03-19_23-27-35/progress.csv")["episode_reward_mean"]
stage_3_results = pd.read_csv("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_rec/PPO_Soccer_15248_00000_0_2026-03-20_16-19-22/progress.csv")["episode_reward_mean"]

# restore="/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_6faba_00000_0_2026-03-28_09-23-21/checkpoint_003000/checkpoint-3000"
# restore="/Users/mathewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_ac4f3_00000_0_2026-03-25_11-44-36/checkpoint_002320/checkpoint-2320"
# restore="/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_7fbbb_00000_0_2026-03-23_23-49-46/checkpoint_000520/checkpoint-520"
# restore="/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_be6b7_00000_0_2026-03-23_17-24-58/checkpoint_000160/checkpoint-160"
stage_4_results_0 = pd.read_csv("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_be6b7_00000_0_2026-03-23_17-24-58/progress.csv")
stage_4_results_1 = pd.read_csv("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_7fbbb_00000_0_2026-03-23_23-49-46/progress.csv")
stage_4_results_2 = pd.read_csv("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_ac4f3_00000_0_2026-03-25_11-44-36/progress.csv")
stage_4_results_3 = pd.read_csv("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_6faba_00000_0_2026-03-28_09-23-21/progress.csv")
# print(stage_4_results_0.columns)
stage_4_results = np.concatenate([
    stage_4_results_0["policy_reward_mean/striker"].to_numpy(),
    stage_4_results_1["policy_reward_mean/striker"].to_numpy(),
    stage_4_results_2["policy_reward_mean/striker"].to_numpy(),
    stage_4_results_3["policy_reward_mean/striker"].to_numpy(),
])
print(stage_2_results.shape)
print(stage_4_results.shape)
fig, axes = plt.subplots(2, 2, figsize=(11,7))
color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

axes[0, 0].plot(stage_1_results, alpha=0.2, color=color)
axes[0, 0].plot(ewa(stage_1_results, 0.1), alpha=1.0, color=color)
axes[0, 0].set_title("Stage 1: Striker Agent")
axes[0, 0].set_xlabel("Iteration")
axes[0, 0].set_ylabel("Avg. Reward")
axes[0, 0].grid(True, alpha=0.3)


axes[0, 1].plot(stage_2_results, alpha=0.2, color=color)
axes[0, 1].plot(ewa(stage_2_results, 0.01), alpha=1.0, color=color)
axes[0, 1].set_title("Stage 2: First Goalie Agent")
axes[0, 1].set_xlabel("Iteration")
axes[0, 1].set_ylabel("Avg. Reward")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(stage_3_results, alpha=0.2, color=color)
axes[1, 0].plot(ewa(stage_3_results, 0.01), alpha=1.0, color=color)
axes[1, 0].set_title("Stage 3: Second Goalie Agent")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].set_ylabel("Avg. Reward")
axes[1, 0].grid(True, alpha=0.3)


axes[1, 1].plot(stage_4_results, alpha=0.2, color=color)
axes[1, 1].plot(ewa(stage_4_results, 0.01), alpha=1.0, color=color)
axes[1, 1].set_title("Stage 4: Full Self-Play")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].set_ylabel("Avg. Reward")
axes[1, 1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("plots/stage_1.pgf")

stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
values = [560, 526, 734, 763]

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(stages, values)

ax.set_xlabel("Stage")
ax.set_ylabel("Points won (out of 1,000)")
ax.set_title("Win Rates vs. Baseline per Stage")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/performance.pgf")