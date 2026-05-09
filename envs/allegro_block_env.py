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

        self.goal = np.array([-0.015, -0.005, 0.035], dtype=np.float32)
        self.lift_goal_z = 0.085

        # Target orientation quaternion.
        # [w, x, y, z]
        self.target_quat = np.array(
            [0.707, 0.0, 0.707, 0.0],
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:16, 0].astype(np.float32),
            high=self.model.actuator_ctrlrange[:16, 1].astype(np.float32),
            dtype=np.float32,
        )

        # qpos + qvel +
        # block_pos +
        # block_quat +
        # target_quat +
        # goal_error +
        # fingertip positions (4x3) +
        # fingertip-to-block vectors (4x3)
        obs_dim = self.model.nq + self.model.nv + 3 + 4 + 4 + 3 + 12 + 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.max_steps = 300
        self.step_count = 0
        self.prev_dist = None

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        block_pos = self.data.xpos[self.block_id].copy()
        block_quat = self.data.xquat[self.block_id].copy()

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
            block_quat,
            self.target_quat,
            goal_error,
            fingertip_positions,
            fingertip_to_block
        ]).astype(np.float32)


    def _quat_to_yaw(self, q):
        # MuJoCo quaternion format: [w, x, y, z]
        w, x, y, z = q
        return float(np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        ))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Start cube away from initial finger penetration.
        # x forward from palm, y centered, z above floor.
        self.data.qpos[16:19] = np.array([
            np.random.uniform(-0.005, 0.005),
            np.random.uniform(-0.010, 0.000),
            0.055
        ], dtype=np.float32)

        # Unit quaternion for block freejoint.
        self.data.qpos[19:23] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Clear freejoint velocity.
        self.data.qvel[16:22] = 0.0

        mujoco.mj_forward(self.model, self.data)


        # Pre-shaped fingertip grasp pose around cube.
        self.data.qpos[:16] = np.array([
            0.25, 1.05, 1.10, 1.00,
            0.05, 1.05, 1.15, 1.05,
            -0.10, 1.00, 1.10, 1.00,
            1.35, 0.85, 0.55, 0.35
        ], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_dist = None
        self.prev_block_pos = self.data.xpos[self.block_id].copy()
        self.prev_block_quat = self.data.xquat[self.block_id].copy()
        self.prev_action = np.zeros(16, dtype=np.float32)

        self.prev_yaw = self._quat_to_yaw(self.data.xquat[self.block_id].copy())
        self.spin_angle = 0.0

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
        mean_tip_dist = float(np.mean(fingertip_dists))
        close_tips = sum(d < 0.065 for d in fingertip_dists)

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

        # -------------------------------------------------
        # Detect finger-specific contacts
        # -------------------------------------------------

        thumb_contact = False
        index_contact = False
        middle_contact = False
        ring_contact = False

        for body in contact_bodies:
            if "th_" in body:
                thumb_contact = True
            if "ff_" in body:
                index_contact = True
            if "mf_" in body:
                middle_contact = True
            if "rf_" in body:
                ring_contact = True

        pinch_grasp = (
            thumb_contact and (
                index_contact
                or middle_contact
                or ring_contact
            )
        )

        lift_height = float(block_pos[2])
        table_height = 0.05

        block_move = float(np.linalg.norm(block_pos - self.prev_block_pos))
        quat_change = float(1.0 - abs(float(np.dot(block_quat, self.prev_block_quat))))

        # Orientation error to target quaternion.
        orientation_error = float(
            1.0 - abs(float(np.dot(block_quat, self.target_quat)))
        )

        # Track cumulative yaw spin for 360-degree rotation.
        yaw = self._quat_to_yaw(block_quat)
        yaw_delta = yaw - self.prev_yaw

        # unwrap angle to [-pi, pi]
        yaw_delta = (yaw_delta + np.pi) % (2.0 * np.pi) - np.pi

        self.spin_angle += abs(float(yaw_delta))

        block_linvel = float(np.linalg.norm(self.data.qvel[16:19]))

        action_delta = float(np.linalg.norm(action - self.prev_action))
        action_mag = float(np.linalg.norm(action))

        # -------------------------------------------------
        # Manipulation states
        # -------------------------------------------------

        touching = contact_count >= 1

        grasping = (
            unique_contacts >= 2
            and finger_dist < 0.06
        )

        lifting = (
            grasping
            and lift_height > (table_height + 0.01)
        )

        stable_hold = (
            lifting
            and block_linvel < 0.15
        )

        reward = 0.0

        # -------------------------------------------------
        # 1. Reach reward
        # -------------------------------------------------

        reward += -8.0 * finger_dist
        reward += -2.0 * mean_tip_dist

        # -------------------------------------------------
        # 2. Touch reward
        # -------------------------------------------------

        if touching:
            reward += 0.5

        # -------------------------------------------------
        # 3. Grasp reward
        # -------------------------------------------------

        if grasping:
            reward += 3.0

        # -------------------------------------------------
        # 3b. Explicit thumb opposition reward
        # -------------------------------------------------

        if thumb_contact:
            reward += 1.0

        if pinch_grasp:
            reward += 8.0

        # -------------------------------------------------
        # 4. Lift reward
        # -------------------------------------------------

        if lifting:
            lift_amount = lift_height - (table_height + 0.01)
            reward += 120.0 * lift_amount

        # -------------------------------------------------
        # 5. Stable hold reward
        # -------------------------------------------------

        if stable_hold:
            reward += 5.0

        # -------------------------------------------------
        # 6. Small movement reward
        # -------------------------------------------------

        reward += 1.0 * block_move

        # -------------------------------------------------
        # 7. Small rotation reward
        # -------------------------------------------------

        reward += 0.5 * quat_change

        # Orientation alignment reward.
        reward += 2.0 * (1.0 - orientation_error)

        # Main Stage 7 reward: spin cube in hand.
        # Spin is rewarded only when fingertips stay close.
        if close_tips >= 2:
            reward += 80.0 * abs(yaw_delta)
            reward += 3.0 * self.spin_angle
        else:
            reward -= 10.0

        # Strongly encourage fingertip contact before spinning.
        reward += 5.0 * close_tips

        # General contact reward is small.
        reward += 0.2 * contact_count
        reward += 0.5 * unique_contacts

        # Strongly prefer two or more fingertip-region contacts.
        if close_tips >= 2:
            reward += 40.0
        if close_tips == 0:
            reward -= 20.0

        # Penalize no-contact spinning / drifting.
        if contact_count == 0:
            reward -= 8.0

        # 7. Goal position progress.
        reward += -20.0 * dist
        if self.prev_dist is not None:
            reward += 150.0 * (self.prev_dist - dist)

        # 8. Smooth action penalties.
        reward += -0.005 * action_mag
        reward += -0.01 * action_delta

        # 9. Drop penalty.
        if lift_height < 0.035:
            reward -= 5.0

        self.prev_dist = dist
        self.prev_block_pos = block_pos.copy()
        self.prev_block_quat = block_quat.copy()
        self.prev_yaw = yaw
        self.prev_action = action.copy()

        success = bool(
            self.spin_angle >= (2.0 * np.pi)
            and unique_contacts >= 2
        )

        if success:
            reward += 50.0

        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "distance": float(dist),
            "finger_dist": float(finger_dist),
            "mean_tip_dist": float(mean_tip_dist),
            "close_tips": int(close_tips),
            "contact_count": int(contact_count),
            "unique_contacts": int(unique_contacts),
            "touching": bool(touching),
            "grasping": bool(grasping),
            "pinch_grasp": bool(pinch_grasp),
            "thumb_contact": bool(thumb_contact),
            "lifting": bool(lifting),
            "stable_hold": bool(stable_hold),
            "success": bool(success),
            "block_pos": block_pos.copy(),
            "block_quat": block_quat.copy(),
            "block_move": float(block_move),
            "quat_change": float(quat_change),
            "orientation_error": float(orientation_error),
            "yaw": float(yaw),
            "yaw_delta": float(yaw_delta),
            "spin_angle": float(self.spin_angle),
            "spin_degrees": float(np.degrees(self.spin_angle)),
            "lift_height": float(block_pos[2]),
            "reward": float(reward),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
