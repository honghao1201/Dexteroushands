"""第一阶段：把竖直手腕移动到球体两侧的预抓取位置。

这一环境只训练手腕的三向平移，手指保持张开。成功后可以把末状态
传给后续的多指夹持环境，避免 PPO 一开始同时学习移动、对指和抬升。
"""
from __future__ import annotations

import pathlib
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE = pathlib.Path(__file__).resolve().parent / "reach_scene.xml"
TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")


class AeroReachEnv(gym.Env):
    """固定球体、竖直掌面下的手腕接近任务。"""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 180, object_position=None, hold_steps: int = 20):
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
        self.grasp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

        # 用关节名称取地址，避免以后调整 XML 顺序时产生隐蔽的索引错误。
        self.slide_joints = {n: self._joint_qpos(n) for n in ("hand_lift", "hand_x", "hand_y")}
        self.slide_actuators = {n: self._actuator(n + "_A") for n in ("hand_lift", "hand_x", "hand_y")}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        # qpos、qvel、控制量、球相对预抓取锚点、五个指尖相对位置和接触安全量。
        obs_size = self.model.nq + self.model.nv + self.model.nu + 3 + 15 + 5 + 5 + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.initial_object_position = np.asarray(object_position if object_position is not None else (0.10, -0.03, -0.03), dtype=np.float64)
        self.object_position = self.initial_object_position.copy()
        self.target_axes = None
        self.target_qpos = np.zeros(3, dtype=np.float64)
        # 球心沿掌面法向留出 25 mm 预抓取间隙，防止整只手直接撞球。
        self.anchor_offset = np.array([0.025, 0.0, 0.0], dtype=np.float64)
        self.anchor_target = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)

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
                c = self.data.contact[cid]
                if {int(c.geom1), int(c.geom2)} == {geom_id, self.object_geom}:
                    contacts[i] += 1.0
        return tips, gaps, contacts

    def _make_target(self):
        """根据张开手指的几何中心计算球体应处的相对位置。"""
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        # 指尖组中心与拇指指尖的中点，是两侧夹持的几何中心。
        tips = self.data.site_xpos[self.tip_sites]
        anchor = 0.5 * (tips[:4].mean(axis=0) + tips[4])
        self.anchor_target = self.object_position + self.anchor_offset
        delta = self.anchor_target - anchor
        # 第一阶段在水平面完成定位，保持当前高度，避免把手掌压向球体或支撑面。
        delta[2] = 0.0
        mount_axes = self.data.xmat[self.mount_body].reshape(3, 3)
        axis_by_name = {name: mount_axes @ self.model.jnt_axis[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in ("hand_lift", "hand_x", "hand_y")}
        self.target_axes = np.column_stack((axis_by_name["hand_lift"], axis_by_name["hand_x"], axis_by_name["hand_y"]))
        self.target_qpos = np.array([0.0, np.dot(delta, axis_by_name["hand_x"]), np.dot(delta, axis_by_name["hand_y"])], dtype=np.float64)
        for i, name in enumerate(("hand_lift", "hand_x", "hand_y")):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.target_qpos[i] = np.clip(self.target_qpos[i], self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1])

    def _observation(self):
        tips, gaps, contacts = self._tip_data()
        object_pos = self.data.xpos[self.object_body]
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        finger_mean = tips[:4].mean(axis=0)
        anchor = 0.5 * (finger_mean + tips[4])
        relative_anchor = (self.object_position + self.anchor_offset) - anchor
        return np.concatenate((
            self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
            relative_anchor, (tips - object_pos).reshape(-1), gaps, contacts,
            palm_axes[:, 1], palm_axes[:, 2],
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        noise = np.array([self.np_random.uniform(-0.015, 0.015), self.np_random.uniform(-0.018, 0.018), 0.0])
        self.object_position = self.initial_object_position + noise
        self.model.body_pos[self.object_body] = self.object_position
        self.model.geom_pos[self.support_geom, 2] = self.object_position[2] - 0.030
        for adr in self.slide_joints.values():
            self.data.qpos[adr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._make_target()
        self.last_action[:] = 0.0
        # 用当前腱长/关节位置作为保持目标，避免接近阶段因手指回弹而误触球体。
        low, high = self.model.actuator_ctrlrange.T
        neutral = np.clip(self.data.actuator_length.copy(), low, high)
        self.last_action[:] = np.clip(2.0 * (neutral - low) / (high - low) - 1.0, -1.0, 1.0)
        for idx in self.slide_actuators.values():
            self.last_action[idx] = 0.0
        self.data.ctrl[:] = neutral
        self.steps = 0
        self.success_hold = 0
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {"target_qpos": self.target_qpos.copy(), "object_pos": self.object_position.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        commanded = self.last_action.copy()
        # 第一阶段只让三个手腕滑轨由策略控制；五指动作固定为张开。
        for name in ("hand_x", "hand_y"):
            idx = self.slide_actuators[name]
            commanded[idx] = action[idx]
        commanded[self.slide_actuators["hand_lift"]] = 0.0
        commanded[:7] = self.last_action[:7]
        self.last_action = 0.75 * self.last_action + 0.25 * commanded
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = low + (self.last_action + 1.0) * 0.5 * (high - low)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        tips, gaps, contacts = self._tip_data()
        object_pos = self.data.xpos[self.object_body]
        anchor = 0.5 * (tips[:4].mean(axis=0) + tips[4])
        relative = (self.object_position + self.anchor_offset) - anchor
        xy_error = float(np.linalg.norm(relative[:2]))
        anchor_error = xy_error
        z_error = abs(float(relative[2]))
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        palm_horizontal = float(np.linalg.norm(palm_axes[:, 1][:2]))
        side_axis = palm_axes[:, 2]
        finger_side = float(np.mean((tips[:4] - object_pos) @ side_axis))
        thumb_side = float((tips[4] - object_pos) @ side_axis)
        opposite_side = bool(finger_side > 0.018 and thumb_side < -0.018)
        penetration = np.maximum(0.0, -gaps)
        action_penalty = float(np.mean(np.square(action - self.last_action)))

        reward = (
            5.0 * np.exp(-anchor_error / 0.035)
            + 2.0 * np.exp(-xy_error / 0.025)
            + 1.5 * np.exp(-z_error / 0.025)
            + 1.5 * palm_horizontal
            + 1.0 * float(opposite_side)
            - 0.02 * action_penalty
            - 120.0 * float(np.sum(penetration))
        )
        pregrasp = xy_error < 0.005 and palm_horizontal > 0.95 and opposite_side and float(np.max(penetration)) < 0.001
        self.success_hold = self.success_hold + 1 if pregrasp else 0
        success = self.success_hold >= self.required_hold_steps
        terminated = bool(success or np.max(penetration) > 0.003)
        truncated = self.steps >= self.max_steps
        info = {"success": success, "pregrasp": pregrasp, "success_hold": self.success_hold, "anchor_error": anchor_error, "xy_error": xy_error, "z_error": z_error, "opposite_side": opposite_side, "palm_horizontal": palm_horizontal, "gaps": gaps.copy(), "contacts": contacts.copy(), "penetration_depth": float(np.max(penetration)), "target_qpos": self.target_qpos.copy()}
        return self._observation(), float(reward), terminated, truncated, info



