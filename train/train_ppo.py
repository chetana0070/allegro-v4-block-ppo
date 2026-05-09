import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_block_env import AllegroBlockEnv


RUN_NAME = "ppo_allegro_grasp_lift_v2"

LOG_DIR = Path("logs") / RUN_NAME
CKPT_DIR = Path("checkpoints") / RUN_NAME
BEST_DIR = Path("checkpoints") / f"{RUN_NAME}_best"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)


def make_env():
    env = AllegroBlockEnv()
    env = Monitor(env, filename=str(LOG_DIR / "monitor.csv"))
    return env


train_env = DummyVecEnv([make_env])
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
)

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
)

eval_env.training = False
eval_env.norm_reward = False

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=str(CKPT_DIR),
    name_prefix=RUN_NAME,
    save_replay_buffer=False,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(BEST_DIR),
    log_path=str(LOG_DIR / "eval"),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.98,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=str(LOG_DIR),
    device="auto",
)

print("===== TRAINING START =====")
print("run:", RUN_NAME)

model.learn(
    total_timesteps=500_000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name=RUN_NAME,
)

model.save(str(CKPT_DIR / "final_model"))
train_env.save(str(CKPT_DIR / "vecnormalize.pkl"))

print("===== TRAINING DONE =====")
print("final model:", CKPT_DIR / "final_model.zip")
print("vecnormalize:", CKPT_DIR / "vecnormalize.pkl")
print("best model dir:", BEST_DIR)
