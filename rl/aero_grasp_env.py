"""Gymnasium environment for learning tendon-space cube pickup."""

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
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_cube")
        self.object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_qpos = self.model.jnt_qposadr[self.object_joint]
        self.grasp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.tip_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")
        ]
        self.support_z = -0.09
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
        return self._observation(), {"object_pos": self.data.xpos[self.object_body].copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        low, high = self.model.actuator_ctrlrange.T
        # Smooth tendon targets to avoid unrealistic cable impulses.
        self.last_action = 0.85 * self.last_action + 0.15 * action
        self.data.ctrl[:] = low + (self.last_action + 1) * 0.5 * (high - low)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        object_pos = self.data.xpos[self.object_body]
        hand_pos = np.mean(self.data.site_xpos[self.tip_sites], axis=0)
        distance = float(np.linalg.norm(object_pos - hand_pos))
        lift = float(object_pos[2] - self.support_z)
        closure = float(np.mean((self.data.ctrl[:4] - low[:4]) / (high[:4] - low[:4])))
        thumb_closure = float(np.mean((self.data.ctrl[5:] - low[5:]) / (high[5:] - low[5:])))
        thumb_contacts, finger_contacts, normal_force = self._object_contacts()
        opposing_contact = min(1.0, thumb_contacts) * min(1.0, finger_contacts / 2.0)
        stable_hold = max(0.0, min(1.0, (lift - 0.015) / 0.04))
        horizontal_speed = float(np.linalg.norm(self.data.qvel[self.object_qpos : self.object_qpos + 2]))
        lift_reward = min(max(0.0, lift), 0.08)
        force_reward = min(normal_force, 100.0)
        reward = (
            1.5 * max(0.0, 1.0 - distance / 0.07)
            + 0.8 * closure
            + 0.8 * thumb_closure
            + 4.0 * opposing_contact
            + 12.0 * lift_reward * opposing_contact
            + 4.0 * stable_hold * opposing_contact
            + 0.03 * force_reward * opposing_contact
            - 0.5 * horizontal_speed
        )
        terminated = bool(object_pos[2] > -0.025 and opposing_contact > 0.5)
        dropped = bool(object_pos[2] < self.support_z - 0.02 or distance > 0.16)
        if terminated:
            reward += 15.0
        if dropped:
            reward -= 10.0
        terminated = terminated or dropped
        truncated = self.steps >= self.max_steps
        info = {"distance": distance, "lift": lift, "thumb_contacts": thumb_contacts, "finger_contacts": finger_contacts,
                "normal_force": normal_force, "success": bool(object_pos[2] > -0.025 and opposing_contact > 0.5)}
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
