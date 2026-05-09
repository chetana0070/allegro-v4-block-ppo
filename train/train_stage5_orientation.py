import os
import sys

sys.path.append(os.path.abspath("."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_block_env import AllegroBlockEnv


LOGDIR = "logs/stage5_orientation"
CKPTDIR = "checkpoints/stage5_orientation"

os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(CKPTDIR, exist_ok=True)


def make_env():
    return AllegroBlockEnv()


env = DummyVecEnv([make_env])

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2,
    tensorboard_log=LOGDIR,
    device="auto",
)

model.learn(
    total_timesteps=300_000,
    progress_bar=True,
)

model.save(f"{CKPTDIR}/final_model")

env.save(f"{CKPTDIR}/vecnormalize.pkl")

print("DONE TRAINING ORIENTATION POLICY")
