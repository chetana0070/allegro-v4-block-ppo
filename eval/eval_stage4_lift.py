import os
import sys
sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_curriculum_env import AllegroBlockEnv


MODEL = "checkpoints/curriculum_stage4_pinch_lift/final_model.zip"
VEC   = "checkpoints/curriculum_stage4_pinch_lift/vecnormalize.pkl"


def make_env():
    return AllegroBlockEnv(stage=4)


env = DummyVecEnv([make_env])
env = VecNormalize.load(VEC, env)

env.training = False
env.norm_reward = False

model = PPO.load(MODEL)

obs = env.reset()

print("===== STAGE 4 LIFT EVAL =====")

for step in range(300):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    info = info[0]

    if step % 10 == 0:
        print(
            step,
            "reward=", round(float(reward[0]), 3),
            "pinch=", info["true_pinch"],
            "lift=", round(info["lift_height"], 4),
            "success=", info["success"]
        )

    if done[0]:
        print("\nDONE AT", step)
        print(info)
        break

print("===== END =====")
