import mujoco
import numpy as np
from envs.allegro_curriculum_env import AllegroBlockEnv

env = AllegroBlockEnv(stage=2)
obs, info = env.reset()

block = env.data.xpos[env.block_id].copy()

print("===== BLOCK =====")
print("block pos:", block)
print("qpos block:", env.data.qpos[16:19])
print("ncon:", env.data.ncon)

print("\n===== FINGERTIPS =====")
for name in ["index_tip", "middle_tip", "ring_tip", "thumb_tip"]:
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, name)
    pos = env.data.site_xpos[sid].copy()
    dist = np.linalg.norm(pos - block)
    print(f"{name:12s} pos={pos} dist_to_block={dist:.4f}")

print("\n===== SUGGESTED PINCH CENTER =====")
index = env.data.site_xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "index_tip")]
thumb = env.data.site_xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "thumb_tip")]

center = 0.5 * (index + thumb)
print("index-thumb midpoint:", center)
print("thumb-index distance:", np.linalg.norm(index - thumb))

print("\nDONE")
