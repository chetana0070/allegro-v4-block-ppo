import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class AllegroBlockEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.model_path = "assets/mujoco_menagerie/wonik_allegro/allegro_block_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.block_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block"
        )

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "index_tip"
        )

        self.goal = np.array([0.06, 0.0, 0.025], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:4, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:4, 1].astype(np.float32),
            dtype=np.float32,
        )

        obs_dim = self.model.nq + self.model.nv + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.max_steps = 150
        self.step_count = 0
        self.prev_dist = None

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        block_pos = self.data.xpos[self.block_id].copy()
        goal_error = self.goal - block_pos

        finger_pos = self.data.site_xpos[self.site_id].copy()
        finger_to_block = block_pos - finger_pos

        return np.concatenate([
            qpos,
            qvel,
            block_pos,
            goal_error,
            finger_pos,
            finger_to_block
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[16:19] = np.array([
            np.random.uniform(0.02, 0.06),
            np.random.uniform(-0.03, 0.03),
            0.025
        ], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)


        self.data.qpos[:16] = np.array([
            0.0, 0.5, 0.6, 0.6,
            0.0, 0.5, 0.6, 0.6,
            0.0, 0.5, 0.6, 0.6,
            0.8, 0.3, 0.2, 0.2
        ], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_dist = None
        self.prev_block_pos = self.data.xpos[self.block_id].copy()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(
            action,
            self.model.actuator_ctrlrange[:4, 0],
            self.model.actuator_ctrlrange[:4, 1],
        )

        self.data.ctrl[:] = 0.0
        self.data.ctrl[:4] = 0.5 * action

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        block_pos = self.data.xpos[self.block_id].copy()
        finger_pos = self.data.site_xpos[self.site_id].copy()

        dist = np.linalg.norm(block_pos - self.goal)
        finger_dist = np.linalg.norm(finger_pos - block_pos)

        reward = -5.0 * dist

        # fingertip approach reward
        reward += -1.0 * finger_dist

        # reward actual block movement
        block_move = np.linalg.norm(block_pos - self.prev_block_pos)
        reward += 10.0 * block_move

        # contact-style bonus
        if finger_dist < 0.04:
            reward += 2.0

        if finger_dist < 0.025:
            reward += 5.0

        # progress toward block/goal
        if self.prev_dist is not None:
            reward += (self.prev_dist - dist) * 20.0

        self.prev_dist = dist
        self.prev_block_pos = block_pos.copy()

        success = bool(dist < 0.035)
        if success:
            reward += 30.0

        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "distance": float(dist),
            "finger_dist": float(finger_dist),
            "success": bool(success),
            "block_pos": block_pos.copy(),
            "reward": float(reward),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
