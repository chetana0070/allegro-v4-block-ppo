import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import imageio
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_curriculum_env import AllegroBlockEnv


MODEL = "checkpoints/final_stage3_true_pinch/final_model.zip"
VEC   = "checkpoints/final_stage3_true_pinch/vecnormalize.pkl"
VIDEO = "videos/final_stage3_true_pinch.mp4"


def make_env():
    return AllegroBlockEnv(stage=3)


raw_env = AllegroBlockEnv(stage=3)

env = DummyVecEnv([make_env])
env = VecNormalize.load(VEC, env)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL, env=env)

renderer = mujoco.Renderer(raw_env.model, height=480, width=640)

obs = env.reset()
frames = []

print("===== FINAL STAGE 3 VIDEO EVAL =====")

for step in range(150):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, infos = env.step(action)

    sim_env = env.venv.envs[0]

    raw_env.data.qpos[:] = sim_env.data.qpos[:]
    raw_env.data.qvel[:] = sim_env.data.qvel[:]
    mujoco.mj_forward(raw_env.model, raw_env.data)

    renderer.update_scene(raw_env.data)
    frames.append(renderer.render())

    info = infos[0]

    print(
        f"step={step:03d} "
        f"reward={info['reward']:.3f} "
        f"thumb={info['thumb_contact']} "
        f"index={info['index_contact']} "
        f"true_pinch={info['true_pinch']} "
        f"hold={info['pinch_hold_steps']} "
        f"success={info['success']}"
    )

    if done[0]:
        print("DONE AT", step)
        break

Path("videos").mkdir(exist_ok=True)
imageio.mimsave(VIDEO, frames, fps=30)

print("saved:", VIDEO)
