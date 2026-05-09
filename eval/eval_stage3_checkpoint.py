import os
import sys
sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_curriculum_env import AllegroBlockEnv


MODEL = "checkpoints/curriculum_stage3_true_pinch/curriculum_stage3_true_pinch_200000_steps.zip"
VEC   = "checkpoints/curriculum_stage3_true_pinch/curriculum_stage3_true_pinch_vecnormalize_200000_steps.pkl"


def make_env():
    return AllegroBlockEnv(stage=3)


env = DummyVecEnv([make_env])
env = VecNormalize.load(VEC, env)

env.training = False
env.norm_reward = False

model = PPO.load(MODEL)

obs = env.reset()

print("===== STAGE 3 EVAL =====")

for step in range(300):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    info = info[0]

    if step % 10 == 0:
        print(
            step,
            "reward=", round(float(reward[0]), 3),
            "thumb=", info["thumb_contact"],
            "pinch=", info["pinch_grasp"],
            "contacts=", info["contact_count"],
            "hold=", info["pinch_hold_steps"],
            "success=", info["success"],
        )

    if done[0]:
        print("\nDONE AT", step)
        print(info)
        break

print("===== END =====")
