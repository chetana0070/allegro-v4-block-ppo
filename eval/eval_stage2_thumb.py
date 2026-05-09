import os
import sys
sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_curriculum_env import AllegroBlockEnv


MODEL = "checkpoints/curriculum_stage2_thumb_scratch/final_model.zip"
VEC   = "checkpoints/curriculum_stage2_thumb_scratch/vecnormalize.pkl"


def make_env():
    return AllegroBlockEnv(stage=2)


env = DummyVecEnv([make_env])
env = VecNormalize.load(VEC, env)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL)

obs = env.reset()

base_env = env.venv.envs[0]

print("===== EVAL START =====")

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
            "success=", info["success"],
            "contacts=", info.get("contact_count", -1),
        )

    if done[0]:
        print("\nEPISODE DONE AT STEP", step)
        print(info)
        break

print("===== EVAL END =====")
