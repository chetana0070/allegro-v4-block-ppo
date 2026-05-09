import numpy as np
import mujoco
from envs.allegro_block_env import AllegroBlockEnv

env = AllegroBlockEnv()
obs, info = env.reset()

model = env.model
data = env.data

block_id = env.block_id
block_pos = data.xpos[block_id].copy()

print("===== RESET STATE =====")
print("block body id:", block_id)
print("block pos:", block_pos)
print("block quat:", data.xquat[block_id])
print("block qpos[16:23]:", data.qpos[16:23])
print("ncon at reset:", data.ncon)

print("\n===== SITES =====")
for name in ["index_tip"]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    print(name, "site id:", sid)
    if sid >= 0:
        pos = data.site_xpos[sid].copy()
        dist = np.linalg.norm(pos - block_pos)
        print("  pos:", pos)
        print("  dist to block:", dist)

print("\n===== BODY POSITIONS NEAR BLOCK =====")
for i in range(model.nbody):
    name = model.body(i).name
    if any(k in name for k in ["ff_", "mf_", "rf_", "th_", "palm", "block"]):
        pos = data.xpos[i].copy()
        dist = np.linalg.norm(pos - block_pos)
        if dist < 0.12:
            print(f"{name:15s} pos={pos} dist={dist:.4f}")

print("\nRESET INSPECTION DONE")
