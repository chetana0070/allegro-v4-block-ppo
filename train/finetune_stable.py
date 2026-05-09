import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.allegro_block_env import AllegroBlockEnv


RUN_NAME = "ppo_allegro_stable_finetune_v4"

LOG_DIR = Path("logs") / RUN_NAME
CKPT_DIR = Path("checkpoints") / RUN_NAME
BEST_DIR = Path("checkpoints") / f"{RUN_NAME}_best"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "checkpoints/ppo_allegro_grasp_lift_v2_best/best_model.zip"
BASE_VECNORM = "checkpoints/ppo_allegro_grasp_lift_v2/vecnormalize.pkl"


def make_env():
    env = AllegroBlockEnv()
    env = Monitor(env, filename=str(LOG_DIR / "monitor.csv"))
    return env


train_env = DummyVecEnv([make_env])
train_env = VecNormalize.load(BASE_VECNORM, train_env)
train_env.training = True
train_env.norm_reward = True

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize.load(BASE_VECNORM, eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load(
    BASE_MODEL,
    env=train_env,
    device="auto",
    learning_rate=1e-4,
    ent_coef=0.003,
)

model.lr_schedule = lambda _: 1e-4
model.ent_coef = 0.003

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=str(CKPT_DIR),
    name_prefix=RUN_NAME,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(BEST_DIR),
    log_path=str(LOG_DIR / "eval"),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

print("===== STABLE FINE-TUNE START =====")
print("base model:", BASE_MODEL)
print("base vecnorm:", BASE_VECNORM)
print("run:", RUN_NAME)

model.learn(
    total_timesteps=150_000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name=RUN_NAME,
    reset_num_timesteps=False,
)

model.save(str(CKPT_DIR / "final_model"))
train_env.save(str(CKPT_DIR / "vecnormalize.pkl"))

print("===== STABLE FINE-TUNE DONE =====")
print("final model:", CKPT_DIR / "final_model.zip")
print("vecnormalize:", CKPT_DIR / "vecnormalize.pkl")
