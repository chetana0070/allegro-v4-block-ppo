import sys, os
sys.path.append(os.path.abspath("."))

import numpy as np
from envs.allegro_block_env import AllegroBlockEnv

positions = [
    [0.025,  0.000, 0.075],
    [0.030,  0.000, 0.075],
    [0.035,  0.000, 0.075],
    [0.040,  0.000, 0.075],
    [0.025, -0.010, 0.075],
    [0.030, -0.010, 0.075],
    [0.035, -0.010, 0.075],
    [0.040, -0.010, 0.075],
]

for pos in positions:
    env = AllegroBlockEnv()
    obs, info = env.reset()

    env.data.qpos[16:19] = np.array(pos)
    env.data.qpos[19:23] = np.array([1,0,0,0])
    env.data.qvel[16:22] = 0
    import mujoco
    mujoco.mj_forward(env.model, env.data)

    block_pos = env.data.xpos[env.block_id].copy()

    dists = []
    for name, sid in zip(env.tip_names, env.tip_ids):
        tip = env.data.site_xpos[sid].copy()
        d = np.linalg.norm(tip - block_pos)
        dists.append((name, round(float(d), 4)))

    print("cube", pos, "actual", np.round(block_pos, 4), "tip_dists", dists)
