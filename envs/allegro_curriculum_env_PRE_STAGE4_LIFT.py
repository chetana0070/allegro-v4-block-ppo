import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class AllegroBlockEnv(gym.Env):
    def __init__(self, stage=1):
        super().__init__()
        self.stage = stage

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
        if self.stage in [2, 3]:
            # Thumb/pinch curriculum spawn: near thumb-index workspace.
            self.data.qpos[16:19] = np.array([
                np.random.uniform(-0.015, 0.000),
                np.random.uniform(-0.040, -0.020),
                0.058
            ], dtype=np.float32)
        else:
            # General object spawn.
            self.data.qpos[16:19] = np.array([
                np.random.uniform(0.100, 0.115),
                np.random.uniform(-0.030, -0.015),
                0.05
            ], dtype=np.float32)

        # Unit quaternion for block freejoint.
        self.data.qpos[19:23] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Clear freejoint velocity.
        self.data.qvel[16:22] = 0.0

        mujoco.mj_forward(self.model, self.data)


        if self.stage == 2:
            # Stage 2: pre-bend thumb toward cube to make thumb contact learnable.
            self.data.qpos[:16] = np.array([
                0.1, 0.8, 0.9, 0.9,
                0.0, 0.9, 1.0, 1.0,
                0.0, 0.9, 1.0, 1.0,
                1.35, 0.95, 0.75, 0.45
            ], dtype=np.float32)
        else:
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
        self.prev_finger_dist = None
        self.touch_hold_steps = 0
        self.pinch_hold_steps = 0

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
        # Curriculum Stage 1: reach + touch
        # -------------------------------------------------
        if self.stage == 1:
            reward = 0.0

            # Strong reach shaping.
            reward += -5.0 * finger_dist

            # Progress toward cube.
            if hasattr(self, "prev_finger_dist") and self.prev_finger_dist is not None:
                reward += 20.0 * (self.prev_finger_dist - finger_dist)

            # Real contact is the main skill.
            if touching:
                reward += 5.0
                self.touch_hold_steps += 1
            else:
                self.touch_hold_steps = 0

            # Reward sustained controlled touch.
            reward += 0.5 * self.touch_hold_steps

            # Avoid violent actions.
            reward += -0.005 * action_mag
            reward += -0.01 * action_delta

            self.prev_finger_dist = finger_dist

            success = bool(
                self.touch_hold_steps >= 10
                and finger_dist < 0.07
            )

            if success:
                reward += 25.0

            terminated = success
            truncated = bool(self.step_count >= self.max_steps)

            self.prev_dist = dist
            self.prev_block_pos = block_pos.copy()
            self.prev_block_quat = block_quat.copy()
            self.prev_action = action.copy()

            info = {
                "distance": float(dist),
                "finger_dist": float(finger_dist),
                "thumb_dist": float(thumb_dist) if self.stage == 2 else -1.0,
                "contact_count": int(contact_count),
                "unique_contacts": int(unique_contacts),
                "touching": bool(touching),
                "touch_hold_steps": int(self.touch_hold_steps),
                "grasping": bool(False),
                "pinch_grasp": bool(False),
                "thumb_contact": bool(thumb_contact),
                "lifting": bool(False),
                "stable_hold": bool(False),
                "success": bool(success),
                "block_pos": block_pos.copy(),
                "block_quat": block_quat.copy(),
                "block_move": float(block_move),
                "quat_change": float(quat_change),
                "lift_height": float(block_pos[2]),
                "reward": float(reward),
            }

            return obs, reward, terminated, truncated, info

        # -------------------------------------------------
        # Curriculum Stage 2: THUMB ONLY
        # -------------------------------------------------
        if self.stage == 2:

            thumb_pos = self.data.site_xpos[self.tip_ids[3]].copy()
            thumb_dist = float(np.linalg.norm(thumb_pos - block_pos))

            reward = 0.0

            # Strong thumb approach shaping.
            reward += -15.0 * thumb_dist

            # Reward ONLY thumb touching.
            if thumb_contact:
                reward += 20.0
                self.pinch_hold_steps += 1
            else:
                self.pinch_hold_steps = 0

            # Sustained thumb contact.
            reward += 2.0 * self.pinch_hold_steps

            # No generic touch reward.
            # No fake-contact penalty here because reset may include incidental contacts.

            # Smooth motion penalties.
            reward += -0.005 * action_mag
            reward += -0.01 * action_delta

            success = bool(self.pinch_hold_steps >= 3)

            self.prev_finger_dist = finger_dist

            success = bool(
                thumb_contact
                and self.pinch_hold_steps >= 3
            )

            if success:
                reward += 40.0

            terminated = success
            truncated = bool(self.step_count >= self.max_steps)

            self.prev_dist = dist
            self.prev_block_pos = block_pos.copy()
            self.prev_block_quat = block_quat.copy()
            self.prev_action = action.copy()

            info = {
                "distance": float(dist),
                "finger_dist": float(finger_dist),
                "contact_count": int(contact_count),
                "unique_contacts": int(unique_contacts),
                "touching": bool(touching),
                "touch_hold_steps": int(getattr(self, "touch_hold_steps", 0)),
                "pinch_hold_steps": int(self.pinch_hold_steps),
                "grasping": bool(grasping),
                "pinch_grasp": bool(pinch_grasp),
                "thumb_contact": bool(thumb_contact),
                "lifting": bool(False),
                "stable_hold": bool(False),
                "success": bool(success),
                "block_pos": block_pos.copy(),
                "block_quat": block_quat.copy(),
                "block_move": float(block_move),
                "quat_change": float(quat_change),
                "lift_height": float(block_pos[2]),
                "reward": float(reward),
            }

            return obs, reward, terminated, truncated, info

        # -------------------------------------------------
        # Curriculum Stage 3: TRUE THUMB + INDEX PINCH
        # -------------------------------------------------
        if self.stage == 3:

            thumb_pos = self.data.site_xpos[self.tip_ids[3]].copy()
            index_pos = self.data.site_xpos[self.tip_ids[0]].copy()

            thumb_dist = float(np.linalg.norm(thumb_pos - block_pos))
            index_dist = float(np.linalg.norm(index_pos - block_pos))

            reward = 0.0

            # Bring both pinch fingers close.
            reward += -8.0 * thumb_dist
            reward += -5.0 * index_dist

            # Thumb is required.
            if thumb_contact:
                reward += 8.0

            # Small cooperative index reward.
            # Only active when thumb is near object.

            if index_contact and thumb_dist < 0.12:
                reward += 2.0

            # True pinch = both sides.
            true_pinch = bool(thumb_contact and index_contact)

            if true_pinch:
                reward += 25.0
                self.pinch_hold_steps += 1
            else:
                self.pinch_hold_steps = 0

            # Sustained pinch.
            reward += 3.0 * self.pinch_hold_steps

            # Keep object from falling off.
            reward += -3.0 * abs(float(block_pos[2]) - 0.058)

            # Smooth action.
            reward += -0.005 * action_mag
            reward += -0.01 * action_delta

            success = bool(self.pinch_hold_steps >= 5)

            if success:
                reward += 50.0

            terminated = success
            truncated = bool(self.step_count >= self.max_steps)

            self.prev_dist = dist
            self.prev_block_pos = block_pos.copy()
            self.prev_block_quat = block_quat.copy()
            self.prev_action = action.copy()

            info = {
                "distance": float(dist),
                "finger_dist": float(finger_dist),
                "thumb_dist": float(thumb_dist),
                "index_dist": float(index_dist),
                "contact_count": int(contact_count),
                "unique_contacts": int(unique_contacts),
                "touching": bool(touching),
                "touch_hold_steps": int(getattr(self, "touch_hold_steps", 0)),
                "pinch_hold_steps": int(self.pinch_hold_steps),
                "grasping": bool(grasping),
                "pinch_grasp": bool(pinch_grasp),
                "true_pinch": bool(true_pinch),
                "thumb_contact": bool(thumb_contact),
                "index_contact": bool(index_contact),
                "lifting": bool(False),
                "stable_hold": bool(False),
                "success": bool(success),
                "block_pos": block_pos.copy(),
                "block_quat": block_quat.copy(),
                "block_move": float(block_move),
                "quat_change": float(quat_change),
                "lift_height": float(block_pos[2]),
                "reward": float(reward),
            }

            return obs, reward, terminated, truncated, info

        # -------------------------------------------------
        # Full task reward starts here
        # -------------------------------------------------

        # -------------------------------------------------
        # 1. Reach reward
        # -------------------------------------------------

        reward += -2.5 * finger_dist

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
            "lift_height": float(block_pos[2]),
            "reward": float(reward),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
