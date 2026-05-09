import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

runs = {
    "Stage 1 Reach/Touch": "logs/curriculum_stage1_reach_touch/monitor.csv",
    "Stage 2 Thumb": "logs/curriculum_stage2_thumb_scratch/monitor.csv",
    "Stage 3 True Pinch": "logs/curriculum_stage3_true_pinch/monitor.csv",
    "Stage 4 Pinch+Lift": "logs/curriculum_stage4_pinch_lift/monitor.csv",
    "Stage 4 Short Lift": "logs/curriculum_stage4_short_lift/monitor.csv",
}

def read_monitor(path):
    # SB3 monitor has first metadata row starting with #
    return pd.read_csv(path, comment="#")

def smooth(x, window=20):
    return x.rolling(window=window, min_periods=1).mean()

all_summary = []

# 1. Reward curve
plt.figure(figsize=(10, 6))
for name, path in runs.items():
    if not os.path.exists(path):
        continue
    df = read_monitor(path)
    df["episode"] = range(1, len(df) + 1)
    plt.plot(df["episode"], smooth(df["r"]), label=name)
    all_summary.append({
        "run": name,
        "episodes": len(df),
        "mean_reward": df["r"].mean(),
        "max_reward": df["r"].max(),
        "mean_ep_len": df["l"].mean(),
        "min_ep_len": df["l"].min(),
    })

plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Reward")
plt.title("Curriculum Learning: Episode Reward vs Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "01_episode_reward_curve.png", dpi=200)

# 2. Episode length curve
plt.figure(figsize=(10, 6))
for name, path in runs.items():
    if not os.path.exists(path):
        continue
    df = read_monitor(path)
    df["episode"] = range(1, len(df) + 1)
    plt.plot(df["episode"], smooth(df["l"]), label=name)

plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Length")
plt.title("Curriculum Learning: Episode Length vs Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "02_episode_length_curve.png", dpi=200)

# 3. Summary bar: max reward
summary = pd.DataFrame(all_summary)

plt.figure(figsize=(10, 6))
plt.bar(summary["run"], summary["max_reward"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Maximum Episode Reward")
plt.title("Best Reward Achieved per Curriculum Stage")
plt.tight_layout()
plt.savefig(OUT / "03_max_reward_by_stage.png", dpi=200)

# 4. Summary bar: mean episode length
plt.figure(figsize=(10, 6))
plt.bar(summary["run"], summary["mean_ep_len"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Mean Episode Length")
plt.title("Mean Episode Length per Curriculum Stage")
plt.tight_layout()
plt.savefig(OUT / "04_mean_episode_length_by_stage.png", dpi=200)

# 5. Checkpoint sweep results from terminal
sweep = pd.DataFrame([
    {"checkpoint": 25000, "successes": 0, "true_pinches": 0, "thumb_hits": 129, "best_hold": 0, "best_reward": 7.08},
    {"checkpoint": 50000, "successes": 0, "true_pinches": 1, "thumb_hits": 184, "best_hold": 1, "best_reward": 37.01},
    {"checkpoint": 75000, "successes": 0, "true_pinches": 0, "thumb_hits": 136, "best_hold": 0, "best_reward": 7.07},
    {"checkpoint": 100000, "successes": 0, "true_pinches": 0, "thumb_hits": 16, "best_hold": 0, "best_reward": 7.07},
    {"checkpoint": 125000, "successes": 1, "true_pinches": 8, "thumb_hits": 270, "best_hold": 5, "best_reward": 98.54},
    {"checkpoint": 150000, "successes": 0, "true_pinches": 0, "thumb_hits": 1, "best_hold": 0, "best_reward": 7.03},
    {"checkpoint": 175000, "successes": 0, "true_pinches": 0, "thumb_hits": 8, "best_hold": 0, "best_reward": 7.04},
    {"checkpoint": 200000, "successes": 1, "true_pinches": 12, "thumb_hits": 176, "best_hold": 5, "best_reward": 98.72},
])

plt.figure(figsize=(10, 6))
plt.plot(sweep["checkpoint"], sweep["true_pinches"], marker="o", label="True Pinches")
plt.plot(sweep["checkpoint"], sweep["successes"], marker="o", label="Successes")
plt.xlabel("Checkpoint Timesteps")
plt.ylabel("Count over 10 Evaluation Episodes")
plt.title("Checkpoint Sweep: True Pinch and Success Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "05_checkpoint_sweep_success.png", dpi=200)

# 6. Thumb hits by checkpoint
plt.figure(figsize=(10, 6))
plt.bar(sweep["checkpoint"].astype(str), sweep["thumb_hits"])
plt.xlabel("Checkpoint Timesteps")
plt.ylabel("Thumb Contact Count")
plt.title("Thumb Contact Emergence Across Checkpoints")
plt.tight_layout()
plt.savefig(OUT / "06_thumb_hits_by_checkpoint.png", dpi=200)

summary.to_csv(OUT / "curriculum_summary.csv", index=False)
sweep.to_csv(OUT / "checkpoint_sweep_summary.csv", index=False)

print("PLOTS SAVED TO:", OUT)
print(summary)
