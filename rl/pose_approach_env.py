"""PoseApproach：先把整只手调整到球体相对的正确姿态。

这一阶段只训练手腕的平移和绕竖直轴旋转，五指保持当前张开构型。
目标是让张开指尖的几何中心落在球心，同时让掌面法向指向球体的
相对接近方向。闭合、夹紧和抬升留给后续阶段处理。
"""
from __future__ import annotations

import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SCENE = pathlib.Path(__file__).resolve().parent / "pose_scene.xml"
TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")


class AeroPoseApproachEnv(gym.Env):
    """球体周围的手腕位姿接近环境。"""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 240, object_position=None, hold_steps: int = 15):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.max_steps = int(max_steps)
        self.required_hold_steps = int(hold_steps)
        self.steps = 0
        self.success_hold = 0

        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_sphere")
        self.support_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "support_surface")
        self.mount_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tetheria_mount")
        self.palm_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.tip_sites = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIP_NAMES]
        self.tip_geoms = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TIP_NAMES]

        self.joint_qpos = {name: self._joint_qpos(name) for name in ("hand_lift", "hand_x", "hand_y", "hand_yaw")}
        self.wrist_actuators = {
            name: self._actuator(name + "_A") for name in ("hand_lift", "hand_x", "hand_y", "hand_yaw")
        }
        self.wrist_order = ("hand_lift", "hand_x", "hand_y", "hand_yaw")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # 观测包含物理状态、球体相对位置、目标法向、姿态误差和碰撞安全量。
        obs_size = self.model.nq + self.model.nv + 3 + 3 + 4 + 6 + 5 + 5 + 3 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        self.initial_object_position = np.asarray(
            object_position if object_position is not None else (0.10, -0.03, -0.03), dtype=np.float64
        )
        self.object_position = self.initial_object_position.copy()
        self.target_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.target_q = np.zeros(4, dtype=np.float64)
        self.command_q = np.zeros(4, dtype=np.float64)
        self.hold_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.prev_anchor_error = 0.0

        # 参考位姿由几何关系给出，RL 动作只允许小幅残差，避免重新探索整条
        # 接近轨迹；这对应 DemoGrasp 的 reference tracking 思路。
        self.residual_ranges = np.array([0.02, 0.03, 0.03, 0.12], dtype=np.float64)

    def _joint_qpos(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[jid])

    def _actuator(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))

    def _tip_data(self):
        tips = self.data.site_xpos[self.tip_sites].copy()
        gaps = np.zeros(5, dtype=np.float32)
        contacts = np.zeros(5, dtype=np.float32)
        for i, geom_id in enumerate(self.tip_geoms):
            gaps[i] = float(mujoco.mj_geomDistance(self.model, self.data, geom_id, self.object_geom, 1.0, np.zeros(6)))
            for cid in range(self.data.ncon):
                contact = self.data.contact[cid]
                if {int(contact.geom1), int(contact.geom2)} == {geom_id, self.object_geom}:
                    contacts[i] += 1.0
        return tips, gaps, contacts

    def _anchor(self, tips: np.ndarray) -> np.ndarray:
        """用拇指和四指指尖的中点近似张开手的抓取中心。"""
        return 0.5 * (tips[:4].mean(axis=0) + tips[4])

    def _mount_axis_world(self, name: str) -> np.ndarray:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mount_axes = self.data.xmat[self.mount_body].reshape(3, 3)
        return mount_axes @ self.model.jnt_axis[jid]

    def _compute_target(self):
        """从零位构型计算球体对应的接近法向、yaw 和腕部平移目标。"""
        # 当前零位张开手的中心决定接近方向：掌面从球体外侧水平朝向球体。
        tips, _, _ = self._tip_data()
        anchor0 = self._anchor(tips)
        horizontal = anchor0[:2] - self.object_position[:2]
        if np.linalg.norm(horizontal) < 1e-5:
            horizontal = np.array([0.0, 1.0], dtype=np.float64)
        horizontal = horizontal / np.linalg.norm(horizontal)
        self.target_normal = np.array([horizontal[0], horizontal[1], 0.0], dtype=np.float64)

        palm_axes0 = self.data.xmat[self.palm_body].reshape(3, 3)
        current_normal = palm_axes0[:, 2].copy()
        current_normal[2] = 0.0
        current_normal /= max(np.linalg.norm(current_normal), 1e-8)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle = float(np.arctan2(np.dot(np.cross(current_normal, self.target_normal), up), np.dot(current_normal, self.target_normal)))
        # hand_yaw 的局部 +Y 映射到世界 -Z，所以关节正方向与世界 yaw 相反。
        yaw = float(np.clip(-angle, -1.90, 1.90))

        # 先临时设置 yaw，读取旋转后的平移轴，再把指尖中心移到球心。
        yaw_adr = self.joint_qpos["hand_yaw"]
        saved_yaw = float(self.data.qpos[yaw_adr])
        self.data.qpos[yaw_adr] = yaw
        mujoco.mj_forward(self.model, self.data)
        tips_yaw, _, _ = self._tip_data()
        anchor_yaw = self._anchor(tips_yaw)
        delta = self.object_position - anchor_yaw
        target = np.zeros(4, dtype=np.float64)
        target[3] = yaw
        # 这些滑轨在 MuJoCo 中的广义坐标实际对应世界 Z/X/Y 平移。
        # 直接使用世界分量可以避免把安装座的固定四元数重复变换一次。
        target[0] = float(delta[2])
        target[1] = float(delta[0])
        target[2] = float(delta[1])
        for i, name in enumerate(("hand_lift", "hand_x", "hand_y")):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            target[i] = float(np.clip(target[i], self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]))
        self.data.qpos[yaw_adr] = saved_yaw
        mujoco.mj_forward(self.model, self.data)
        self.target_q = target

    def _orientation_error(self) -> tuple[float, float, float]:
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        normal = palm_axes[:, 2].copy()
        normal[2] = 0.0
        normal /= max(np.linalg.norm(normal), 1e-8)
        angle = float(np.arccos(np.clip(np.dot(normal, self.target_normal), -1.0, 1.0)))
        vertical = float(abs(np.dot(palm_axes[:, 0], np.array([0.0, 0.0, 1.0]))))
        return angle, vertical, float(np.dot(normal, self.target_normal))

    def _observation(self):
        tips, gaps, contacts = self._tip_data()
        object_pos = self.data.xpos[self.object_body]
        anchor = self._anchor(tips)
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        angle, vertical, normal_dot = self._orientation_error()
        return np.concatenate(
            (
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                object_pos - anchor,
                self.target_normal,
                self.target_q - self.command_q,
                palm_axes[:, 1],
                palm_axes[:, 2],
                gaps,
                contacts,
                np.array([angle, vertical, normal_dot], dtype=np.float64),
                np.array([self.steps / max(self.max_steps, 1)], dtype=np.float64),
            )
        ).astype(np.float32)

    def _set_controls(self):
        self.data.ctrl[:] = self.hold_ctrl
        for i, name in enumerate(self.wrist_order):
            self.data.ctrl[self.wrist_actuators[name]] = self.command_q[i]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        noise = np.array(
            [self.np_random.uniform(-0.025, 0.025), self.np_random.uniform(-0.035, 0.035), 0.0],
            dtype=np.float64,
        )
        self.object_position = self.initial_object_position + noise
        self.model.body_pos[self.object_body] = self.object_position
        self.model.geom_pos[self.support_geom, 2] = self.object_position[2] - 0.085
        self.data.qpos[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # 先记录张开手的执行器目标，PoseApproach 中五指全部保持此目标。
        self.hold_ctrl[:] = self.data.actuator_length.copy()
        self.command_q[:] = 0.0
        self._compute_target()
        self.last_action[:] = 0.0
        self.steps = 0
        self.success_hold = 0
        self._set_controls()
        mujoco.mj_forward(self.model, self.data)
        self.prev_anchor_error = float(np.linalg.norm(self.object_position - self._anchor(self._tip_data()[0])))
        return self._observation(), {"target_q": self.target_q.copy(), "target_normal": self.target_normal.copy(), "object_pos": self.object_position.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        # 解析参考目标负责主要移动，策略只学习小范围的避碰/平滑修正。
        desired_q = self.target_q + action.astype(np.float64) * self.residual_ranges
        self.command_q = 0.75 * self.command_q + 0.25 * desired_q
        for i, name in enumerate(self.wrist_order):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.command_q[i] = np.clip(self.command_q[i], self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1])
        self.last_action = action.copy()
        self._set_controls()
        # 接近阶段使用运动学腕部目标，避免高自由度手模型的关节阻尼把姿态
        # 训练变成一个额外的动力学控制问题；后续接触/抬升阶段仍使用真实步进。
        for i, name in enumerate(self.wrist_order):
            self.data.qpos[self.joint_qpos[name]] = self.command_q[i]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        tips, gaps, contacts = self._tip_data()
        anchor_error = float(np.linalg.norm(self.object_position - self._anchor(tips)))
        angle_error, palm_vertical, normal_dot = self._orientation_error()
        penetration = np.maximum(0.0, -gaps)
        action_penalty = float(np.mean(np.square(action)))
        target_error = float(np.linalg.norm(self.target_q - self.command_q))
        progress = self.prev_anchor_error - anchor_error
        self.prev_anchor_error = anchor_error

        reward = (
            10.0 * np.exp(-anchor_error / 0.025)
            + 5.0 * np.exp(-angle_error / 0.20)
            + 1.5 * palm_vertical
            + 1.0 * max(normal_dot, 0.0)
            + 5.0 * np.exp(-target_error / 0.08)
            + 20.0 * progress
            - 0.02 * action_penalty
            - 120.0 * float(np.sum(penetration))
        )
        pose_ok = anchor_error < 0.008 and angle_error < 0.10 and palm_vertical > 0.98 and float(np.max(penetration)) < 0.001
        self.success_hold = self.success_hold + 1 if pose_ok else 0
        success = self.success_hold >= self.required_hold_steps
        # 穿透只作为惩罚，不在第一步立刻结束回合；否则随机探索很难获得
        # “先到预抓取位姿再保持”的长时序信号。
        terminated = bool(success)
        truncated = self.steps >= self.max_steps
        info = {
            "success": success,
            "pose_ok": pose_ok,
            "success_hold": self.success_hold,
            "anchor_error": anchor_error,
            "angle_error": angle_error,
            "palm_vertical": palm_vertical,
            "normal_dot": normal_dot,
            "target_error": target_error,
            "command_q": self.command_q.copy(),
            "gaps": gaps.copy(),
            "contacts": contacts.copy(),
            "penetration_depth": float(np.max(penetration)),
            "target_q": self.target_q.copy(),
            "target_normal": self.target_normal.copy(),
        }
        return self._observation(), float(reward), terminated, truncated, info
