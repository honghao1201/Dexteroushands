"""Shadow Hand 从上方接近球体的预抓取环境。

本阶段只训练整只手的预抓取位姿，不闭合手指，也不抬升物体。策略先把掌面
转为朝下，再把手整体沿世界 Z 轴下降到球体上方的安全预抓取高度。
"""
from __future__ import annotations

import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SCENE = pathlib.Path(__file__).resolve().parent.parent / "models" / "shadow_hand" / "scene_topdown.xml"


class ShadowTopDownApproachEnv(gym.Env):
    """Shadow Hand 顶部接近阶段：掌面朝下、从上向下到达预抓取位姿。"""

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
        self.palm_body = self._body("rh_palm")
        self.anchor_site = self._site("approach_anchor")
        self.root_order = ("hand_lift", "hand_x", "hand_y", "hand_pitch", "hand_yaw")
        self.root_qpos = {name: self._joint_qpos(name) for name in self.root_order}
        self.root_actuators = {name: self._actuator(name + "_A") for name in self.root_order}
        self.hand_geom_ids = [
            gid for gid in range(self.model.ngeom) if gid not in (self.object_geom, self.support_geom)
        ]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        # qpos/qvel、目标锚点误差、目标掌面法向、目标关节残差、掌面轴、间隙和姿态指标。
        obs_size = self.model.nq + self.model.nv + 3 + 3 + 5 + 6 + 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        self.initial_object_position = np.asarray(
            object_position if object_position is not None else (0.10, -0.03, -0.03), dtype=np.float64
        )
        self.object_position = self.initial_object_position.copy()
        # 锚点位于球心上方 100 mm，末端指尖接近但不穿透球体。
        self.anchor_offset = np.array([0.0, 0.0, 0.10], dtype=np.float64)
        self.anchor_target = self.object_position + self.anchor_offset
        self.target_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.target_q = np.zeros(5, dtype=np.float64)
        self.command_q = np.zeros(5, dtype=np.float64)
        self.hold_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.residual_ranges = np.array([0.025, 0.025, 0.025, 0.12, 0.16], dtype=np.float64)
        self.last_action = np.zeros(5, dtype=np.float32)
        self.prev_anchor_error = 0.0
        self.prev_pose_error = 0.0
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

    def _pose_metrics(self):
        # Shadow Hand 的 palm_axes 第三列是掌面法向；顶下抓取要求它接近世界 -Z。
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        normal = palm_axes[:, 2].copy()
        normal_dot = float(np.dot(normal, self.target_normal))
        angle = float(np.arccos(np.clip(normal_dot, -1.0, 1.0)))
        palm_down = float(abs(np.dot(normal, np.array([0.0, 0.0, 1.0]))))
        return palm_axes, angle, palm_down, normal_dot

    def _contact_metrics(self):
        min_gap = 1.0
        contacts = 0.0
        for geom_id in self.hand_geom_ids:
            min_gap = min(
                min_gap,
                float(mujoco.mj_geomDistance(self.model, self.data, geom_id, self.object_geom, 1.0, np.zeros(6))),
            )
        for cid in range(self.data.ncon):
            contact = self.data.contact[cid]
            if self.object_geom in (int(contact.geom1), int(contact.geom2)):
                contacts += 1.0
        return min_gap, contacts

    def _compute_target(self):
        # 先将姿态设为掌面朝下，再用解析锚点误差求三个平移关节的目标值。
        pitch = -0.5 * np.pi
        yaw = 0.0
        self.data.qpos[self.root_qpos["hand_pitch"]] = pitch
        self.data.qpos[self.root_qpos["hand_yaw"]] = yaw
        mujoco.mj_forward(self.model, self.data)
        delta = self.anchor_target - self._anchor()
        target = np.array([delta[2], delta[0], delta[1], pitch, yaw], dtype=np.float64)
        for i, name in enumerate(self.root_order):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            target[i] = np.clip(target[i], self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1])
        self.target_q = target

    def _set_controls(self):
        self.data.ctrl[:] = self.hold_ctrl
        for name in self.root_order:
            self.data.ctrl[self.root_actuators[name]] = self.command_q[self.root_order.index(name)]

    def _observation(self):
        anchor = self._anchor()
        palm_axes, angle, palm_down, normal_dot = self._pose_metrics()
        min_gap, contacts = self._contact_metrics()
        return np.concatenate(
            (
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.anchor_target - anchor,
                self.target_normal,
                self.target_q - self.command_q,
                palm_axes[:, 1],
                palm_axes[:, 2],
                np.array(
                    [min_gap, contacts, angle, palm_down, normal_dot, self.steps / max(self.max_steps, 1)],
                    dtype=np.float64,
                ),
            )
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        noise = np.array(
            [self.np_random.uniform(-0.025, 0.025), self.np_random.uniform(-0.035, 0.035), 0.0],
            dtype=np.float64,
        )
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
        _, pose_angle, _, _ = self._pose_metrics()
        self.prev_pose_error = pose_angle
        return self._observation(), {
            "target_q": self.target_q.copy(),
            "target_normal": self.target_normal.copy(),
            "object_pos": self.object_position.copy(),
        }

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
        _, angle_error, palm_down, normal_dot = self._pose_metrics()
        min_gap, contacts = self._contact_metrics()
        penetration = max(0.0, -min_gap)
        target_error = float(np.linalg.norm(self.target_q - self.command_q))
        anchor_progress = self.prev_anchor_error - anchor_error
        pose_progress = self.prev_pose_error - angle_error
        self.prev_anchor_error = anchor_error
        self.prev_pose_error = angle_error

        # 参考分阶段预抓取方法：位置、姿态、进度和安全间隙共同提供密集反馈。
        clearance_error = abs(min_gap - 0.003) if min_gap >= 0.0 else abs(min_gap)
        reward = (
            12.0 * np.exp(-anchor_error / 0.035)
            + 8.0 * np.exp(-angle_error / 0.25)
            + 3.0 * palm_down
            + 4.0 * np.exp(-target_error / 0.15)
            + 22.0 * anchor_progress
            + 4.0 * pose_progress
            - 0.5 * clearance_error
            - 180.0 * penetration
            - 0.02 * float(np.mean(np.square(action)))
        )
        pose_ok = (
            anchor_error < 0.008
            and angle_error < 0.10
            and palm_down > 0.98
            and min_gap > -1e-4
            and min_gap < 0.02
        )
        self.success_hold = self.success_hold + 1 if pose_ok else 0
        success = self.success_hold >= self.required_hold_steps
        terminated = bool(success)
        truncated = self.steps >= self.max_steps
        info = {
            "success": success,
            "pose_ok": pose_ok,
            "success_hold": self.success_hold,
            "anchor_error": anchor_error,
            "angle_error": angle_error,
            "palm_down": palm_down,
            "normal_dot": normal_dot,
            "target_error": target_error,
            "min_gap": min_gap,
            "contacts": contacts,
            "penetration_depth": penetration,
            "topdown_height": float(self.anchor_target[2] - self.object_position[2]),
            "target_q": self.target_q.copy(),
            "command_q": self.command_q.copy(),
            # DummyVecEnv 在 done 后会自动 reset；回放脚本用这两个快照保留真正的终止帧。
            "terminal_qpos": self.data.qpos.copy(),
            "terminal_qvel": self.data.qvel.copy(),
        }
        return self._observation(), float(reward), terminated, truncated, info
