
import os, sys
sys.path.append(os.path.abspath("."))

import imageio
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.allegro_block_env import AllegroBlockEnv

VIDEO_PATH = "videos/stage5_orientation_eval.mp4"
os.makedirs("videos", exist_ok=True)

env = DummyVecEnv([lambda: AllegroBlockEnv()])
env = VecNormalize.load("checkpoints/stage5_orientation/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("checkpoints/stage5_orientation/final_model.zip", env=env)

obs = env.reset()
base_env = env.venv.envs[0]

renderer = mujoco.Renderer(base_env.model, height=480, width=640)
frames = []
total_reward = 0.0

for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += float(reward[0])

    renderer.update_scene(base_env.data)
    frames.append(renderer.render())

    if step % 25 == 0:
        print(
            "step", step,
            "reward", round(float(reward[0]), 3),
            "orient_err", round(float(info[0]["orientation_error"]), 3), "spin_deg", round(float(info[0]["spin_degrees"]), 1),
            "lift", round(float(info[0]["lift_height"]), 3),
            "close_tips", info[0]["close_tips"], "contacts", info[0]["contact_count"], "unique", info[0]["unique_contacts"], "success", info[0]["success"],
        )

    if done[0]:
        print("done at step", step)
        break

imageio.mimsave(VIDEO_PATH, frames, fps=30)
print("VIDEO SAVED:", VIDEO_PATH)
print("TOTAL REWARD:", total_reward)

