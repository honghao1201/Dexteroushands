"""Shadow Hand 的 PoseApproach 阶段。

Shadow Hand 保留 24 个手部关节；本阶段另外给整手增加三向平移和一个
绕世界竖直轴的 yaw。手指关节保持张开，PPO 只学习解析参考位姿附近的
小残差，先解决手掌相对物体的接近方向。
"""
from __future__ import annotations

import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SCENE = pathlib.Path(__file__).resolve().parent.parent / "models" / "shadow_hand" / "scene_pose.xml"


class ShadowPoseApproachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 180, object_position=None, hold_steps: int = 15):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.max_steps = int(max_steps)
        self.required_hold_steps = int(hold_steps)

        self.object_body = self._body("grasp_object")
        self.object_geom = self._geom("grasp_sphere")
        self.support_geom = self._geom("support_surface")
        self.mount_body = self._body("rh_forearm")
        self.palm_body = self._body("rh_palm")
        self.anchor_site = self._site("approach_anchor")
        self.root_order = ("hand_lift", "hand_x", "hand_y", "hand_yaw")
        self.root_qpos = {name: self._joint_qpos(name) for name in self.root_order}
        self.root_actuators = {name: self._actuator(name + "_A") for name in self.root_order}
        self.hand_geom_ids = [gid for gid in range(self.model.ngeom) if gid not in (self.object_geom, self.support_geom)]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # qpos/qvel、球体相对锚点、目标法向、目标残差、掌部方向、最近间隙、
        # 接触数量、方向/竖直度/法向相似度和时间比例。
        obs_size = self.model.nq + self.model.nv + 3 + 3 + 4 + 6 + 1 + 1 + 3 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        self.initial_object_position = np.asarray(
            object_position if object_position is not None else (0.10, -0.03, -0.03), dtype=np.float64
        )
        self.object_position = self.initial_object_position.copy()
        # 预抓取时把球心放在四指与拇指张开间隙的几何中心，
        # 相比掌部锚点沿四指方向留出约 30 mm，避免球体穿过近侧指节。
        self.anchor_offset = np.array([0.0, 0.030, 0.0], dtype=np.float64)
        self.anchor_target = self.object_position + self.anchor_offset
        self.target_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.target_q = np.zeros(4, dtype=np.float64)
        self.command_q = np.zeros(4, dtype=np.float64)
        self.hold_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.residual_ranges = np.array([0.02, 0.03, 0.03, 0.12], dtype=np.float64)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.prev_anchor_error = 0.0
        self.steps = 0
        self.success_hold = 0

    def _body(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))

    def _geom(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))

    def _site(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name))

    def _joint_qpos(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[jid])

    def _actuator(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))

    def _anchor(self) -> np.ndarray:
        return self.data.site_xpos[self.anchor_site].copy()

    def _palm_metrics(self):
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        normal = palm_axes[:, 2].copy()
        normal[2] = 0.0
        normal /= max(np.linalg.norm(normal), 1e-8)
        angle = float(np.arccos(np.clip(np.dot(normal, self.target_normal), -1.0, 1.0)))
        vertical = float(abs(np.dot(palm_axes[:, 1], np.array([0.0, 0.0, 1.0]))))
        normal_dot = float(np.dot(normal, self.target_normal))
        return palm_axes, angle, vertical, normal_dot

    def _contact_metrics(self):
        min_gap = 1.0
        contacts = 0.0
        for geom_id in self.hand_geom_ids:
            min_gap = min(min_gap, float(mujoco.mj_geomDistance(self.model, self.data, geom_id, self.object_geom, 1.0, np.zeros(6))))
        for cid in range(self.data.ncon):
            c = self.data.contact[cid]
            if self.object_geom in (int(c.geom1), int(c.geom2)):
                contacts += 1.0
        return min_gap, contacts

    def _compute_target(self):
        anchor0 = self._anchor()
        horizontal = anchor0[:2] - self.object_position[:2]
        if np.linalg.norm(horizontal) < 1e-6:
            horizontal = np.array([1.0, 0.0], dtype=np.float64)
        horizontal /= np.linalg.norm(horizontal)
        self.target_normal = np.array([horizontal[0], horizontal[1], 0.0], dtype=np.float64)

        _, _, _, _ = self._palm_metrics()
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        current_normal = palm_axes[:, 2].copy()
        current_normal[2] = 0.0
        current_normal /= max(np.linalg.norm(current_normal), 1e-8)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        yaw = float(np.clip(np.arctan2(np.dot(np.cross(current_normal, self.target_normal), up), np.dot(current_normal, self.target_normal)), -1.9, 1.9))

        yaw_adr = self.root_qpos["hand_yaw"]
        saved = float(self.data.qpos[yaw_adr])
        self.data.qpos[yaw_adr] = yaw
        mujoco.mj_forward(self.model, self.data)
        delta = self.anchor_target - self._anchor()
        # 根部滑轨的实际世界轴：hand_lift=Z，hand_x=X，hand_y=Y。
        target = np.array([delta[2], delta[0], delta[1], yaw], dtype=np.float64)
        for i, name in enumerate(self.root_order):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            target[i] = np.clip(target[i], self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1])
        self.data.qpos[yaw_adr] = saved
        mujoco.mj_forward(self.model, self.data)
        self.target_q = target

    def _set_controls(self):
        self.data.ctrl[:] = self.hold_ctrl
        for name in self.root_order:
            self.data.ctrl[self.root_actuators[name]] = self.command_q[self.root_order.index(name)]

    def _observation(self):
        anchor = self._anchor()
        palm_axes, angle, vertical, normal_dot = self._palm_metrics()
        min_gap, contacts = self._contact_metrics()
        return np.concatenate(
            (
                self.data.qpos.copy(), self.data.qvel.copy(),
                self.anchor_target - anchor, self.target_normal,
                self.target_q - self.command_q, palm_axes[:, 1], palm_axes[:, 2],
                np.array([min_gap, contacts, angle, vertical, normal_dot, self.steps / max(self.max_steps, 1)], dtype=np.float64),
            )
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        noise = np.array([self.np_random.uniform(-0.025, 0.025), self.np_random.uniform(-0.035, 0.035), 0.0], dtype=np.float64)
        self.object_position = self.initial_object_position + noise
        self.anchor_target = self.object_position + self.anchor_offset
        self.model.body_pos[self.object_body] = self.object_position
        self.model.geom_pos[self.support_geom, 2] = self.object_position[2] - 0.085
        self.data.qpos[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.hold_ctrl[:] = self.data.actuator_length.copy()
        self.command_q[:] = 0.0
        self._compute_target()
        self.last_action[:] = 0.0
        self.steps = 0
        self.success_hold = 0
        self._set_controls()
        mujoco.mj_forward(self.model, self.data)
        self.prev_anchor_error = float(np.linalg.norm(self.anchor_target - self._anchor()))
        return self._observation(), {"target_q": self.target_q.copy(), "target_normal": self.target_normal.copy(), "object_pos": self.object_position.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        desired_q = self.target_q + action.astype(np.float64) * self.residual_ranges
        self.command_q = 0.75 * self.command_q + 0.25 * desired_q
        for i, name in enumerate(self.root_order):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.command_q[i] = np.clip(self.command_q[i], self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1])
        self.last_action = action.copy()
        self._set_controls()
        for i, name in enumerate(self.root_order):
            self.data.qpos[self.root_qpos[name]] = self.command_q[i]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        anchor_error = float(np.linalg.norm(self.anchor_target - self._anchor()))
        _, angle_error, palm_vertical, normal_dot = self._palm_metrics()
        min_gap, contacts = self._contact_metrics()
        penetration = max(0.0, -min_gap)
        target_error = float(np.linalg.norm(self.target_q - self.command_q))
        progress = self.prev_anchor_error - anchor_error
        self.prev_anchor_error = anchor_error
        reward = (
            10.0 * np.exp(-anchor_error / 0.025)
            + 5.0 * np.exp(-angle_error / 0.20)
            + 1.5 * palm_vertical + max(normal_dot, 0.0)
            + 5.0 * np.exp(-target_error / 0.08) + 20.0 * progress
            - 0.02 * float(np.mean(np.square(action))) - 120.0 * penetration
        )
        pose_ok = anchor_error < 0.008 and angle_error < 0.10 and palm_vertical > 0.98 and penetration < 0.001
        self.success_hold = self.success_hold + 1 if pose_ok else 0
        success = self.success_hold >= self.required_hold_steps
        terminated = bool(success)
        truncated = self.steps >= self.max_steps
        info = {
            "success": success, "pose_ok": pose_ok, "success_hold": self.success_hold,
            "anchor_error": anchor_error, "angle_error": angle_error,
            "palm_vertical": palm_vertical, "normal_dot": normal_dot,
            "target_error": target_error, "min_gap": min_gap, "contacts": contacts,
            "penetration_depth": penetration, "target_q": self.target_q.copy(),
            "command_q": self.command_q.copy(), "target_normal": self.target_normal.copy(),
        }
        return self._observation(), float(reward), terminated, truncated, info
