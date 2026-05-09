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

        self.tip_names = [
            "index_tip",
            "middle_tip",
            "ring_tip",
            "thumb_tip",
        ]

        self.tip_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name
            )
            for name in self.tip_names
        ]

        self.goal = np.array([0.12, 0.0, 0.05], dtype=np.float32)
        self.lift_goal_z = 0.085

        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:16, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:16, 1].astype(np.float32),
            dtype=np.float32,
        )

        # qpos + qvel +
        # block_pos +
        # goal_error +
        # fingertip positions (4x3) +
        # fingertip-to-block vectors (4x3)
        obs_dim = self.model.nq + self.model.nv + 3 + 3 + 12 + 12
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

        fingertip_positions = []
        fingertip_to_block = []

        for sid in self.tip_ids:
            pos = self.data.site_xpos[sid].copy()
            fingertip_positions.append(pos)
            fingertip_to_block.append(block_pos - pos)

        fingertip_positions = np.concatenate(fingertip_positions)
        fingertip_to_block = np.concatenate(fingertip_to_block)

        return np.concatenate([
            qpos,
            qvel,
            block_pos,
            goal_error,
            fingertip_positions,
            fingertip_to_block
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Start cube away from initial finger penetration.
        # x forward from palm, y centered, z above floor.
        self.data.qpos[16:19] = np.array([
            np.random.uniform(0.080, 0.090),
            np.random.uniform(-0.010, 0.010),
            0.05
        ], dtype=np.float32)

        # Unit quaternion for block freejoint.
        self.data.qpos[19:23] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Clear freejoint velocity.
        self.data.qvel[16:22] = 0.0

        mujoco.mj_forward(self.model, self.data)


        self.data.qpos[:16] = np.array([
            0.1, 0.8, 0.9, 0.9,
            0.0, 0.9, 1.0, 1.0,
            0.0, 0.9, 1.0, 1.0,
            1.2, 0.6, 0.3, 0.2
        ], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_dist = None
        self.prev_block_pos = self.data.xpos[self.block_id].copy()
        self.prev_block_quat = self.data.xquat[self.block_id].copy()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(
            action,
            self.model.actuator_ctrlrange[:16, 0],
            self.model.actuator_ctrlrange[:16, 1],
        )

        self.data.ctrl[:] = 0.0
        self.data.ctrl[:16] = 0.5 * action

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        block_pos = self.data.xpos[self.block_id].copy()
        block_quat = self.data.xquat[self.block_id].copy()
        fingertip_positions = [
            self.data.site_xpos[sid].copy()
            for sid in self.tip_ids
        ]

        fingertip_dists = [
            np.linalg.norm(pos - block_pos)
            for pos in fingertip_positions
        ]

        finger_dist = float(np.min(fingertip_dists))

        dist = np.linalg.norm(block_pos - self.goal)

        # contact reward
        contact_count = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]

            g1 = self.model.geom_bodyid[con.geom1]
            g2 = self.model.geom_bodyid[con.geom2]

            b1 = self.model.body(g1).name
            b2 = self.model.body(g2).name

            pair = f"{b1} {b2}"

            if (
                ("block" in pair)
                and
                (
                    "link" in pair
                    or "finger" in pair
                    or "palm" in pair
                )
            ):
                contact_count += 1

        # lift-first dexterous reward
        lift_height = block_pos[2]

        reward = 0.0

        # reach the cube
        reward += -2.0 * finger_dist

        # move cube upward
        reward += 40.0 * max(0.0, lift_height - 0.04)

        # keep cube near palm/target area, but weakly
        reward += -1.0 * dist

        # reward actual block movement
        block_move = np.linalg.norm(block_pos - self.prev_block_pos)
        reward += 5.0 * block_move

        # reward actual block rotation slightly
        quat_change = 1.0 - abs(float(np.dot(block_quat, self.prev_block_quat)))
        reward += 5.0 * quat_change

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
        self.prev_block_quat = block_quat.copy()

        success = bool(block_pos[2] > self.lift_goal_z)

        if success:
            reward += 50.0

        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "distance": float(dist),
            "finger_dist": float(finger_dist),
            "success": bool(success),
            "block_pos": block_pos.copy(),
            "block_quat": block_quat.copy(),
            "block_move": float(block_move),
            "quat_change": float(quat_change),
            "lift_height": float(block_pos[2]),
            "reward": float(reward),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
