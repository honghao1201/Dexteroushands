"""Gymnasium environment for vertical spherical-object grasp and lift."""

from __future__ import annotations

import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE = pathlib.Path(__file__).resolve().parent / "grasp_scene.xml"


class AeroGraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.steps = 0
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_sphere")
        self.object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_qpos = self.model.jnt_qposadr[self.object_joint]
        self.hand_lift_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_lift")
        self.hand_lift_qpos = self.model.jnt_qposadr[self.hand_lift_joint]
        self.grasp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.tip_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")
        ]
        self.support_z = -0.09
        self.required_hold_steps = 20
        self.hold_steps = 0
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_size = self.model.nq + self.model.nv + self.model.nu + 6 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

    def _observation(self) -> np.ndarray:
        object_pos = self.data.xpos[self.object_body]
        object_vel = self.data.qvel[self.object_qpos : self.object_qpos + 6]
        return np.concatenate((self.data.qpos, self.data.qvel, self.data.ctrl, object_pos, object_vel)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Keep the object near the palm while randomizing its lateral position.
        self.data.qpos[self.object_qpos : self.object_qpos + 3] = np.array(
            [0.13 + self.np_random.uniform(-0.008, 0.008), -0.015 + self.np_random.uniform(-0.008, 0.008), -0.064]
        )
        self.data.qvel[:] = 0
        self.last_action[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.hold_steps = 0
        return self._observation(), {"object_pos": self.data.xpos[self.object_body].copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        low, high = self.model.actuator_ctrlrange.T
        # Smooth tendon targets to avoid unrealistic cable impulses.
        self.last_action = 0.85 * self.last_action + 0.15 * action
        # The lift channel is one-sided: -1/0 means stay on the support, +1 raises
        # the complete hand. This prevents an untrained zero-mean policy from
        # immediately launching the hand and sphere upward.
        target_action = self.last_action.copy()
        target_action[7] = max(0.0, float(target_action[7]))
        self.data.ctrl[:] = low + (target_action + 1) * 0.5 * (high - low)
        self.data.ctrl[7] = low[7] + target_action[7] * (high[7] - low[7])
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        object_pos = self.data.xpos[self.object_body]
        hand_pos = np.mean(self.data.site_xpos[self.tip_sites], axis=0)
        distance = float(np.linalg.norm(object_pos - hand_pos))
        lift = float(object_pos[2] - self.support_z)
        hand_lift = float(self.data.qpos[self.hand_lift_qpos])
        closure = float(np.mean((self.data.ctrl[:4] - low[:4]) / (high[:4] - low[:4])))
        thumb_closure = float(np.mean((self.data.ctrl[5:7] - low[5:7]) / (high[5:7] - low[5:7])))
        lift_command = float(np.clip((self.data.ctrl[7] - low[7]) / (high[7] - low[7]), 0.0, 1.0))
        thumb_contacts, finger_contacts, normal_force = self._object_contacts()
        opposing_contact = min(1.0, thumb_contacts) * min(1.0, finger_contacts / 2.0)
        horizontal_speed = float(np.linalg.norm(self.data.qvel[self.object_qpos : self.object_qpos + 2]))
        vertical_speed = float(abs(self.data.qvel[self.object_qpos + 2]))
        angular_speed = float(np.linalg.norm(self.data.qvel[self.object_qpos + 3 : self.object_qpos + 6]))
        # Success requires a raised hand, opposing contacts and sustained low-speed hold.
        lifted = lift > 0.065
        stable = bool(
            lifted
            and hand_lift > 0.035
            and lift_command > 0.25
            and opposing_contact > 0.5
            and normal_force > 0.05
            and horizontal_speed < 0.08
            and vertical_speed < 0.12
            and angular_speed < 2.0
        )
        self.hold_steps = self.hold_steps + 1 if stable else 0
        success = self.hold_steps >= self.required_hold_steps
        stable_hold = min(1.0, self.hold_steps / self.required_hold_steps)
        lift_progress = min(1.0, max(0.0, hand_lift) / 0.12)
        lift_reward = min(max(0.0, lift), 0.08)
        force_reward = min(normal_force, 100.0)
        reward = (
            1.5 * max(0.0, 1.0 - distance / 0.07)
            + 0.8 * closure
            + 0.8 * thumb_closure
            + 4.0 * opposing_contact
            + 12.0 * lift_reward * opposing_contact * lift_command
            + 4.0 * opposing_contact * lift_progress
            + 1.0 * opposing_contact * lift_command
            + 4.0 * stable_hold * opposing_contact
            + 0.03 * force_reward * opposing_contact
            - 0.5 * horizontal_speed
        )
        terminated = bool(success)
        # Keep an episode alive while the policy learns to approach and close.
        # Declare a drop only after the ball is clearly below the support or far away.
        dropped = bool(object_pos[2] < self.support_z - 0.07 or distance > 0.35)
        if success:
            reward += 15.0
        if dropped:
            reward -= 10.0
        terminated = terminated or dropped
        truncated = self.steps >= self.max_steps
        info = {
            "distance": distance,
            "lift": lift,
            "hand_lift": hand_lift,
            "thumb_contacts": thumb_contacts,
            "finger_contacts": finger_contacts,
            "opposing_contact": opposing_contact,
            "normal_force": normal_force,
            "lift_command": lift_command,
            "horizontal_speed": horizontal_speed,
            "vertical_speed": vertical_speed,
            "angular_speed": angular_speed,
            "hold_steps": self.hold_steps,
            "stable": stable,
            "success": success,
        }
        return self._observation(), reward, terminated, truncated, info

    def _object_contacts(self) -> tuple[int, int, float]:
        thumb_contacts = 0
        finger_contacts = 0
        normal_force = 0.0
        force = np.zeros(6, dtype=np.float64)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 != self.object_geom and contact.geom2 != self.object_geom:
                continue
            other_geom = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
            other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(other_geom)) or ""
            other_body = int(self.model.geom_bodyid[other_geom])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other_body) or ""
            contact_name = f"{other_name} {body_name}".lower()
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal_force += max(0.0, float(force[0]))
            if "thumb" in contact_name:
                thumb_contacts += 1
            elif any(finger in contact_name for finger in ("index", "middle", "ring", "pinky")):
                finger_contacts += 1
        return thumb_contacts, finger_contacts, normal_force
