import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import imageio
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_block_env import AllegroBlockEnv


RUN_NAME = "ppo_allegro_grasp_lift_v2"

MODEL_PATH = Path("checkpoints") / RUN_NAME / "ppo_allegro_grasp_lift_v2_125000_steps.zip"
VECNORM_PATH = Path("checkpoints") / RUN_NAME / "ppo_allegro_grasp_lift_v2_vecnormalize_125000_steps.pkl"

# Use best model if final model does not exist yet.
BEST_MODEL_PATH = Path("checkpoints") / f"{RUN_NAME}_best" / "best_model.zip"

VIDEO_PATH = Path("videos") / f"{RUN_NAME}_eval.mp4"
VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_env():
    return AllegroBlockEnv()


raw_env = AllegroBlockEnv()

vec_env = DummyVecEnv([make_env])

if VECNORM_PATH.exists():
    vec_env = VecNormalize.load(str(VECNORM_PATH), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print("Loaded VecNormalize:", VECNORM_PATH)
else:
    print("VecNormalize not found yet. Using raw env.")

if MODEL_PATH.exists():
    model_path = MODEL_PATH
elif BEST_MODEL_PATH.exists():
    model_path = BEST_MODEL_PATH
else:
    raise FileNotFoundError(
        f"No model found. Tried {MODEL_PATH} and {BEST_MODEL_PATH}"
    )

model = PPO.load(str(model_path), env=vec_env)
print("Loaded model:", model_path)

obs = vec_env.reset()

renderer = mujoco.Renderer(raw_env.model, height=480, width=640)

frames = []
episode_reward = 0.0

# Use raw env only for rendering/extra state sync by stepping separately is bad.
# So for now render from vec_env internal env.
render_env = vec_env.venv.envs[0]

for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    info0 = info[0]
    episode_reward += float(reward[0])

    renderer.update_scene(render_env.data)
    frames.append(renderer.render())

    print(
        f"step={step:03d} "
        f"reward={float(reward[0]):.3f} "
        f"z={info0.get('lift_height', 0):.4f} "
        f"dist={info0.get('distance', 0):.4f} "
        f"finger={info0.get('finger_dist', 0):.4f} "
        f"contacts={info0.get('contact_count', 0)} "
        f"unique={info0.get('unique_contacts', 0)} "
        f"touch={info0.get('touching', False)} "
        f"grasp={info0.get('grasping', False)} "
        f"lift={info0.get('lifting', False)} "
        f"success={info0.get('success', False)}"
    )

    if bool(done[0]):
        break

imageio.mimsave(str(VIDEO_PATH), frames, fps=30)

print("\n===== EVAL DONE =====")
print("model:", model_path)
print("episode_reward:", episode_reward)
print("video:", VIDEO_PATH)
