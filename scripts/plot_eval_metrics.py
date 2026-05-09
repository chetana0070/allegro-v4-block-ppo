import os
import sys
sys.path.append(os.path.abspath("."))

import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.allegro_block_env import AllegroBlockEnv

OUT = "plots"
os.makedirs(OUT, exist_ok=True)

env = DummyVecEnv([lambda: AllegroBlockEnv()])
env = VecNormalize.load("checkpoints/stage5_orientation/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("checkpoints/stage5_orientation/final_model.zip", env=env)

obs = env.reset()
rows = []

for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    info = info[0]

    rows.append({
        "step": step,
        "reward": float(reward[0]),
        "spin_degrees": float(info.get("spin_degrees", 0)),
        "orientation_error": float(info.get("orientation_error", 0)),
        "contact_count": int(info.get("contact_count", 0)),
        "unique_contacts": int(info.get("unique_contacts", 0)),
        "close_tips": int(info.get("close_tips", 0)),
        "lift_height": float(info.get("lift_height", 0)),
        "finger_dist": float(info.get("finger_dist", 0)),
    })

    if done[0]:
        break

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT, "eval_metrics.csv")
df.to_csv(csv_path, index=False)
print("Saved:", csv_path)

def plot_col(col, title, ylabel, filename):
    plt.figure(figsize=(9, 5))
    plt.plot(df["step"], df[col])
    plt.title(title)
    plt.xlabel("Evaluation step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)

plot_col("reward", "Evaluation Reward Over Time", "Reward", "eval_reward_curve.png")
plot_col("spin_degrees", "Object Reorientation Progress", "Spin / reorientation measure", "eval_spin_curve.png")
plot_col("orientation_error", "Orientation Error Over Time", "Orientation error", "eval_orientation_error_curve.png")
plot_col("contact_count", "Hand-Object Contact Count", "Contact count", "eval_contact_count_curve.png")
plot_col("unique_contacts", "Unique Contact Bodies", "Unique contacts", "eval_unique_contacts_curve.png")
plot_col("close_tips", "Fingertip Proximity Count", "Close fingertips", "eval_close_tips_curve.png")
plot_col("finger_dist", "Minimum Fingertip Distance", "Distance", "eval_fingertip_distance_curve.png")
plot_col("lift_height", "Cube Height During Evaluation", "Height", "eval_lift_height_curve.png")

print("Done.")
