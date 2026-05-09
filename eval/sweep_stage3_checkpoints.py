import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.allegro_curriculum_env import AllegroBlockEnv


CKPT_DIR = Path("checkpoints/curriculum_stage3_true_pinch")

steps = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]

def make_env():
    return AllegroBlockEnv(stage=3)


print("===== STAGE 3 CHECKPOINT SWEEP =====")

results = []

for step in steps:
    model_path = CKPT_DIR / f"curriculum_stage3_true_pinch_{step}_steps.zip"
    vec_path = CKPT_DIR / f"curriculum_stage3_true_pinch_vecnormalize_{step}_steps.pkl"

    if not model_path.exists() or not vec_path.exists():
        continue

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)

    successes = 0
    true_pinches = 0
    thumb_hits = 0
    best_hold = 0
    best_reward = -1e9

    for ep in range(10):
        obs = env.reset()

        for t in range(150):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            info = infos[0]

            best_reward = max(best_reward, float(info.get("reward", reward[0])))
            best_hold = max(best_hold, int(info.get("pinch_hold_steps", 0)))

            if info.get("thumb_contact", False):
                thumb_hits += 1
            if info.get("true_pinch", False):
                true_pinches += 1
            if info.get("success", False):
                successes += 1

            if done[0]:
                break

    results.append((step, successes, true_pinches, thumb_hits, best_hold, best_reward))

    print(
        f"step={step:6d} "
        f"successes={successes:3d} "
        f"true_pinches={true_pinches:3d} "
        f"thumb_hits={thumb_hits:3d} "
        f"best_hold={best_hold:2d} "
        f"best_reward={best_reward:.2f}"
    )

print("\n===== BEST BY SUCCESS, PINCH, HOLD =====")
best = sorted(results, key=lambda x: (x[1], x[2], x[4], x[5]), reverse=True)
print(best[:5])
