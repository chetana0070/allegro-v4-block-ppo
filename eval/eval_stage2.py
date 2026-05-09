import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import imageio
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_curriculum_env import AllegroBlockEnv


RUN_NAME = "curriculum_stage2_pinch"

MODEL_PATH = "checkpoints/curriculum_stage2_pinch/final_model.zip"
VEC_PATH   = "checkpoints/curriculum_stage2_pinch/vecnormalize.pkl"

VIDEO_PATH = "videos/curriculum_stage2_pinch.mp4"


def make_env():
    return AllegroBlockEnv(stage=2)


raw_env = AllegroBlockEnv(stage=2)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load(VEC_PATH, vec_env)

vec_env.training = False
vec_env.norm_reward = False

model = PPO.load(MODEL_PATH, env=vec_env)

renderer = mujoco.Renderer(
    raw_env.model,
    height=480,
    width=640,
)

obs = vec_env.reset()

frames = []

print("===== STAGE 2 EVAL =====")

for step in range(300):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = vec_env.step(action)

    raw_env.data.qpos[:] = vec_env.venv.envs[0].data.qpos[:]
    raw_env.data.qvel[:] = vec_env.venv.envs[0].data.qvel[:]

    mujoco.mj_forward(raw_env.model, raw_env.data)

    renderer.update_scene(raw_env.data)
    pixels = renderer.render()

    frames.append(pixels)

    i = info[0]

    print(
        f"step={step} "
        f"reward={reward[0]:.3f} "
        f"pinch={i['pinch_grasp']} "
        f"hold={i['pinch_hold_steps']} "
        f"touch={i['touching']}"
    )

    if done[0]:
        print("episode done")
        break

imageio.mimsave(
    VIDEO_PATH,
    frames,
    fps=30,
)

print("\nDONE")
print("saved:", VIDEO_PATH)
