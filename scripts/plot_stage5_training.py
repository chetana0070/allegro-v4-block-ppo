import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = Path("logs/stage5_orientation")
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

event_files = sorted(LOG_ROOT.rglob("events.out.tfevents*"))
print("Found event files:", len(event_files))

rows = []

for event_file in event_files:
    run = event_file.parent.name
    ea = EventAccumulator(str(event_file.parent))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    for tag in tags:
        for e in ea.Scalars(tag):
            rows.append({
                "run": run,
                "tag": tag,
                "step": e.step,
                "value": e.value,
            })

df = pd.DataFrame(rows)
csv_path = OUT / "stage5_tensorboard_scalars.csv"
df.to_csv(csv_path, index=False)
print("Saved:", csv_path)

def plot_tag(tag, filename, title, ylabel):
    sub = df[df["tag"] == tag]
    if sub.empty:
        print("Missing tag:", tag)
        return

    plt.figure(figsize=(9, 5))
    for run, g in sub.groupby("run"):
        g = g.sort_values("step")
        plt.plot(g["step"], g["value"], label=run, alpha=0.85)

    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = OUT / filename
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)

plot_tag("rollout/ep_rew_mean", "reward_curve.png", "Mean Episode Reward During PPO Training", "Mean episode reward")
plot_tag("rollout/ep_len_mean", "episode_length_curve.png", "Mean Episode Length During PPO Training", "Mean episode length")
plot_tag("train/value_loss", "value_loss_curve.png", "PPO Value Loss", "Value loss")
plot_tag("train/policy_gradient_loss", "policy_gradient_loss_curve.png", "PPO Policy Gradient Loss", "Policy gradient loss")
plot_tag("train/approx_kl", "approx_kl_curve.png", "PPO Approximate KL Divergence", "Approx KL")
plot_tag("train/entropy_loss", "entropy_loss_curve.png", "PPO Entropy Loss", "Entropy loss")
plot_tag("train/clip_fraction", "clip_fraction_curve.png", "PPO Clip Fraction", "Clip fraction")
plot_tag("train/explained_variance", "explained_variance_curve.png", "PPO Explained Variance", "Explained variance")

print("Done.")
