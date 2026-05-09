import imageio
import numpy as np
import mujoco

from stable_baselines3 import PPO
from envs.allegro_block_env import AllegroBlockEnv


env = AllegroBlockEnv()

model = PPO.load(
    "checkpoints/ppo_allegro_block_v3_16dof"
)

obs, info = env.reset()

renderer = mujoco.Renderer(env.model, height=480, width=640)

frames = []

episode_reward = 0

for step in range(300):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    renderer.update_scene(env.data)
    pixels = renderer.render()

    frames.append(pixels)

    print(
        f"step={step} "
        f"reward={reward:.3f} "
        f"dist={info['distance']:.4f} "
        f"move={info['block_move']:.5f} "
        f"rot={info['quat_change']:.6f}"
    )

    if terminated or truncated:
        break

video_path = "videos/allegro_rotation_eval.mp4"

imageio.mimsave(video_path, frames, fps=30)

print("\nDONE")
print("episode reward:", episode_reward)
print("saved:", video_path)
