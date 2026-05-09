import os
import sys

sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.allegro_block_env import AllegroBlockEnv


env = AllegroBlockEnv()
check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=128,
    gamma=0.98,
    gae_lambda=0.95,
    ent_coef=0.02,
    tensorboard_log="logs/ppo_allegro_lift_v1",
)

model.learn(total_timesteps=200_000)

model.save("checkpoints/ppo_allegro_lift_v1")

print("training done")
print("saved: checkpoints/ppo_allegro_lift_v1.zip")
