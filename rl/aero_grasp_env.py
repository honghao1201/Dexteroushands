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
        self.object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_qpos = self.model.jnt_qposadr[self.object_joint]
        self.grasp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
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
            [0.085 + self.np_random.uniform(-0.015, 0.015), self.np_random.uniform(-0.02, 0.02), -0.06]
        )
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._observation(), {"object_pos": self.data.xpos[self.object_body].copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = low + (action + 1) * 0.5 * (high - low)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        object_pos = self.data.xpos[self.object_body]
        hand_pos = self.data.site_xpos[self.grasp_site]
        distance = float(np.linalg.norm(object_pos - hand_pos))
        lift = float(object_pos[2] + 0.06)
        contact = min(1.0, max(0.0, 1.0 - distance / 0.08))
        reward = 2.0 * contact + 8.0 * max(0.0, lift)
        terminated = bool(object_pos[2] > 0.015)
        if terminated:
            reward += 10.0
        truncated = self.steps >= self.max_steps
        info = {"distance": distance, "lift": lift, "success": terminated}
        return self._observation(), reward, terminated, truncated, info
