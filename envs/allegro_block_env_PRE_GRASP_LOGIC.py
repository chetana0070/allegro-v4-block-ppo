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
        self.prev_action = np.zeros(16, dtype=np.float32)

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

        # Count real block-hand contacts.
        contact_count = 0
        contact_bodies = set()

        for i in range(self.data.ncon):
            con = self.data.contact[i]

            b1 = self.model.body(self.model.geom_bodyid[con.geom1]).name
            b2 = self.model.body(self.model.geom_bodyid[con.geom2]).name
            pair = f"{b1} {b2}"

            is_block_contact = "block" in pair
            is_hand_contact = any(
                key in pair
                for key in [
                    "ff_", "mf_", "rf_", "th_",
                    "palm"
                ]
            )

            if is_block_contact and is_hand_contact:
                contact_count += 1
                if b1 != "block":
                    contact_bodies.add(b1)
                if b2 != "block":
                    contact_bodies.add(b2)

        unique_contacts = len(contact_bodies)

        lift_height = float(block_pos[2])
        table_height = 0.05

        block_move = float(np.linalg.norm(block_pos - self.prev_block_pos))
        quat_change = float(1.0 - abs(float(np.dot(block_quat, self.prev_block_quat))))

        action_delta = float(np.linalg.norm(action - self.prev_action))
        action_mag = float(np.linalg.norm(action))

        reward = 0.0

        # 1. Reach: closest fingertip should approach cube.
        reward += -3.0 * finger_dist

        # 2. Real touch reward.
        if contact_count > 0:
            reward += 2.0

        # 3. Multi-finger grasp reward.
        reward += 1.0 * min(unique_contacts, 3)

        # 4. Lift only above table height.
        lift_amount = max(0.0, lift_height - table_height)
        reward += 80.0 * lift_amount

        # 5. Small reward for moving cube, but not dominant.
        reward += 2.0 * block_move

        # 6. Small reward for actual rotation.
        reward += 1.0 * quat_change

        # 7. Goal position progress.
        if self.prev_dist is not None:
            reward += 10.0 * (self.prev_dist - dist)

        # 8. Smooth action penalties.
        reward += -0.005 * action_mag
        reward += -0.01 * action_delta

        # 9. Drop penalty.
        if lift_height < 0.035:
            reward -= 5.0

        self.prev_dist = dist
        self.prev_block_pos = block_pos.copy()
        self.prev_block_quat = block_quat.copy()
        self.prev_action = action.copy()

        success = bool(
            lift_height > self.lift_goal_z
            and contact_count > 0
        )

        if success:
            reward += 50.0

        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "distance": float(dist),
            "finger_dist": float(finger_dist),
            "contact_count": int(contact_count),
            "unique_contacts": int(unique_contacts),
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
