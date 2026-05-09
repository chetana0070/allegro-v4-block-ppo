import os, sys
sys.path.append(os.getcwd())

import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.allegro_curriculum_env import AllegroBlockEnv

MODEL = "checkpoints/curriculum_stage2_thumb_scratch/final_model.zip"
VEC = "checkpoints/curriculum_stage2_thumb_scratch/vecnormalize.pkl"

def make_env():
    return AllegroBlockEnv(stage=2)

env = DummyVecEnv([make_env])
env = VecNormalize.load(VEC, env)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL, env=env)

obs = env.reset()
real_env = env.venv.envs[0]

for step in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, infos = env.step(action)
    info = infos[0]

    print("\nSTEP", step)
    print("reward:", float(reward[0]))
    print("thumb:", info["thumb_contact"], "pinch:", info["pinch_grasp"], "success:", info["success"])
    print("qpos thumb:", real_env.data.qpos[12:16])
    print("ncon:", real_env.data.ncon)

    for i in range(real_env.data.ncon):
        c = real_env.data.contact[i]
        b1 = real_env.model.body(real_env.model.geom_bodyid[c.geom1]).name
        b2 = real_env.model.body(real_env.model.geom_bodyid[c.geom2]).name
        print(" ", i, b1, b2, "dist", round(c.dist, 5))

    if done[0]:
        break
