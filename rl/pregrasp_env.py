"""第二阶段 Pregrasp：锁定手腕，学习拇指与四指的预抓取构型。"""
from __future__ import annotations
import pathlib
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE = pathlib.Path(__file__).resolve().parent / "reach_scene.xml"
TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")


class AeroPregraspEnv(gym.Env):
    """把球放在拇指和四指之间，训练指腹接近但避免穿透。"""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 220, object_position=None, hold_steps: int = 25):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.max_steps = int(max_steps)
        self.required_hold_steps = int(hold_steps)
        self.thumb_phase_steps = 80
        self.steps = 0
        self.success_hold = 0
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_sphere")
        self.support_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "support_surface")
        self.mount_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tetheria_mount")
        self.palm_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.tip_sites = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIP_NAMES]
        self.tip_geoms = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TIP_NAMES]
        self.slide_qpos = {n: self._joint_qpos(n) for n in ("hand_lift", "hand_x", "hand_y")}
        self.slide_actuators = {n: self._actuator(n + "_A") for n in ("hand_lift", "hand_x", "hand_y")}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        # 状态包括机械状态、指尖相对球体位置、预抓取目标、距离和接触安全量。
        obs_size = self.model.nq + self.model.nv + self.model.nu + 15 + 15 + 5 + 5 + 3 + 3 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.initial_object_position = np.asarray(object_position if object_position is not None else (0.10, -0.03, -0.03), dtype=np.float64)
        self.object_position = self.initial_object_position.copy()
        self.anchor_offset = np.array([0.025, 0.030, 0.0], dtype=np.float64)
        self.target_tips = np.zeros((5, 3), dtype=np.float64)
        # 不同指根的几何范围不同，使用可实现的安全间隙目标。
        self.target_gaps = np.array([0.006, 0.006, 0.006, 0.025, 0.006], dtype=np.float64)
        self.target_qpos = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.prev_gap_error = 0.0
        self.prev_thumb_error = 0.0
        self.prev_finger_error = 0.0

    def _joint_qpos(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[jid])

    def _actuator(self, name):
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))

    def _tip_data(self):
        tips = self.data.site_xpos[self.tip_sites].copy()
        gaps = np.zeros(5, dtype=np.float32)
        contacts = np.zeros(5, dtype=np.float32)
        for i, gid in enumerate(self.tip_geoms):
            gaps[i] = float(mujoco.mj_geomDistance(self.model, self.data, gid, self.object_geom, 1.0, np.zeros(6)))
            for cid in range(self.data.ncon):
                c = self.data.contact[cid]
                if {int(c.geom1), int(c.geom2)} == {gid, self.object_geom}:
                    contacts[i] += 1.0
        return tips, gaps, contacts

    def _calculate_targets(self):
        tips, _, _ = self._tip_data()
        object_pos = self.object_position
        # 球体两侧各留 5 mm 安全间隙；保留每根手指的高度，避免手掌从下方托球。
        for i in range(4):
            self.target_tips[i] = object_pos + np.array([0.025, 0.030, tips[i, 2] - object_pos[2]])
        self.target_tips[4] = object_pos + np.array([0.025, -0.030, 0.0])
        anchor = 0.5 * (tips[:4].mean(axis=0) + tips[4])
        desired_anchor = object_pos + self.anchor_offset
        delta = desired_anchor - anchor
        mount_axes = self.data.xmat[self.mount_body].reshape(3, 3)
        axes = {}
        for name in ("hand_lift", "hand_x", "hand_y"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            axes[name] = mount_axes @ self.model.jnt_axis[jid]
        self.target_qpos[:] = (0.0, np.dot(delta, axes["hand_x"]), np.dot(delta, axes["hand_y"]))
        for i, name in enumerate(("hand_lift", "hand_x", "hand_y")):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.target_qpos[i] = np.clip(self.target_qpos[i], self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1])

    def _lock_wrist(self):
        # Pregrasp 阶段不再让策略改变手腕位置；仅保持三条滑轨目标。
        for i, name in enumerate(("hand_lift", "hand_x", "hand_y")):
            self.data.qpos[self.slide_qpos[name]] = self.target_qpos[i]
            self.data.qvel[self.slide_qpos[name]] = 0.0

    def _observation(self):
        tips, gaps, contacts = self._tip_data()
        object_pos = self.data.xpos[self.object_body]
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        return np.concatenate((
            self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
            (tips - object_pos).reshape(-1), (self.target_tips - object_pos).reshape(-1),
            gaps, contacts, palm_axes[:, 1], palm_axes[:, 2], np.array([self.steps / max(1, self.max_steps)]),
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.object_position = self.initial_object_position + np.array([
            self.np_random.uniform(-0.012, 0.012), self.np_random.uniform(-0.015, 0.015), 0.0
        ])
        self.model.body_pos[self.object_body] = self.object_position
        self.model.geom_pos[self.support_geom, 2] = self.object_position[2] - 0.09
        mujoco.mj_forward(self.model, self.data)
        self._calculate_targets()
        self._lock_wrist()
        mujoco.mj_forward(self.model, self.data)
        low, high = self.model.actuator_ctrlrange.T
        neutral = np.clip(self.data.actuator_length.copy(), low, high)
        # 以当前腱长作为张开初始值，防止 reset 后拇指突然扫过球体。
        self.last_action[:] = np.clip(2.0 * (neutral - low) / (high - low) - 1.0, -1.0, 1.0)
        for idx in self.slide_actuators.values():
            self.last_action[idx] = float(np.clip(2.0 * (self.target_qpos[0] - low[idx]) / (high[idx] - low[idx]) - 1.0, -1.0, 1.0)) if idx == self.slide_actuators["hand_lift"] else self.last_action[idx]
        for idx, q in zip((self.slide_actuators["hand_x"], self.slide_actuators["hand_y"]), self.target_qpos[1:]):
            self.last_action[idx] = float(np.clip(2.0 * (q - low[idx]) / (high[idx] - low[idx]) - 1.0, -1.0, 1.0))
        self.data.ctrl[:] = low + (self.last_action + 1.0) * 0.5 * (high - low)
        self.steps = 0
        self.success_hold = 0
        initial_gaps = self._tip_data()[1]
        self.prev_gap_error = float(np.mean(np.abs(initial_gaps - self.target_gaps)))
        self.prev_thumb_error = float(abs(initial_gaps[4] - self.target_gaps[4]))
        self.prev_finger_error = float(np.mean(np.abs(initial_gaps[:4] - self.target_gaps[:4])))
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {"object_pos": self.object_position.copy(), "target_tips": self.target_tips.copy(), "target_gaps": self.target_gaps.copy(), "target_qpos": self.target_qpos.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        commanded = self.last_action.copy()
        # 采用拇指优先的门控课程：前 80 步只训练拇指，避免四指奖励掩盖拇指进度。
        if self.steps < self.thumb_phase_steps:
            # 残差策略叠加轻微拇指闭合先验；策略仍可用正残差减小或停止闭合。
            commanded[:4] = self.last_action[:4]
            commanded[4:7] = np.clip(self.last_action[4:7] + 0.10 * (-0.35 + action[4:7]), -1.0, 1.0)
        else:
            # 拇指到位后再给四指一个较小的闭合先验，避免一步冲过球面。
            commanded[:4] = np.clip(self.last_action[:4] + 0.10 * (-0.20 + action[:4]), -1.0, 1.0)
            commanded[4:7] = np.clip(self.last_action[4:7] + 0.10 * action[4:7], -1.0, 1.0)
        commanded[self.slide_actuators["hand_lift"]] = self.last_action[self.slide_actuators["hand_lift"]]
        commanded[self.slide_actuators["hand_x"]] = self.last_action[self.slide_actuators["hand_x"]]
        commanded[self.slide_actuators["hand_y"]] = self.last_action[self.slide_actuators["hand_y"]]
        self.last_action = 0.80 * self.last_action + 0.20 * commanded
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = low + (self.last_action + 1.0) * 0.5 * (high - low)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self._lock_wrist()
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        tips, gaps, contacts = self._tip_data()
        object_pos = self.data.xpos[self.object_body]
        tip_error = np.linalg.norm(tips - self.target_tips, axis=1)
        mean_error = float(np.mean(tip_error))
        side_axis = self.data.xmat[self.palm_body].reshape(3, 3)[:, 2]
        side_values = (tips - object_pos) @ side_axis
        finger_side = float(np.mean(side_values[:4]))
        thumb_side = float(side_values[4])
        # 至少三根手指在 +Y 侧，拇指在 -Y 侧；允许最外侧小指保留安全距离。
        finger_side_score = float(np.mean(np.clip((side_values[:4] - 0.012) / 0.030, 0.0, 1.0)))
        thumb_side_score = float(np.clip((-thumb_side - 0.012) / 0.030, 0.0, 1.0))
        opposite_side = bool(np.sum(side_values[:4] > 0.012) >= 3 and thumb_side < -0.012)
        palm_normal = self.data.xmat[self.palm_body].reshape(3, 3)[:, 1]
        palm_horizontal = float(np.linalg.norm(palm_normal[:2]))
        penetration = np.maximum(0.0, -gaps)
        gap_error = np.abs(gaps - self.target_gaps)
        mean_gap_error = float(np.mean(gap_error))
        thumb_gap_error = float(gap_error[4])
        finger_gap_error = float(np.mean(gap_error[:4]))
        gap_progress = self.prev_gap_error - mean_gap_error
        thumb_progress = self.prev_thumb_error - thumb_gap_error
        finger_progress = self.prev_finger_error - finger_gap_error
        self.prev_gap_error = mean_gap_error
        self.prev_thumb_error = thumb_gap_error
        self.prev_finger_error = finger_gap_error
        finger_gap_score = float(np.mean(np.exp(-gap_error[:4] / 0.012)))
        thumb_gap_score = float(np.exp(-thumb_gap_error / 0.012))
        # 拇指先导门控：拇指侧向位置越正确，四指闭合奖励才越完整。
        finger_gate = 0.25 + 0.75 * thumb_side_score
        # 拇指单独使用更高权重，避免四指的小幅改善掩盖拇指没有移动。
        reward = (
            2.0 * finger_side_score
            + 6.0 * thumb_side_score
            + 1.5 * finger_gate * finger_gap_score
            + 6.0 * thumb_gap_score
            + 1.0 * palm_horizontal
            + 1.0 * gap_progress
            + 3.0 * thumb_progress
            + 1.0 * finger_progress
            - 0.02 * float(np.mean(np.square(action - self.last_action)))
            - 100.0 * float(np.sum(penetration))
        )
        finger_gap_ok = int(np.sum(gap_error[:4] < 0.012)) >= 3
        thumb_gap_ok = bool(gap_error[4] < 0.012)
        pregrasp = bool(finger_gap_ok and thumb_gap_ok and opposite_side and palm_horizontal > 0.95 and np.max(penetration) < 0.001)
        self.success_hold = self.success_hold + 1 if pregrasp else 0
        success = self.success_hold >= self.required_hold_steps
        terminated = bool(success or np.max(penetration) > 0.003)
        truncated = self.steps >= self.max_steps
        info = {"success": success, "pregrasp": pregrasp, "success_hold": self.success_hold, "tip_error": tip_error.copy(), "mean_tip_error": mean_error, "gap_error": gap_error.copy(), "mean_gap_error": float(np.mean(gap_error)), "finger_side_score": finger_side_score, "thumb_side_score": thumb_side_score, "finger_gap_score": finger_gap_score, "thumb_gap_score": thumb_gap_score, "gap_progress": gap_progress, "thumb_progress": thumb_progress, "finger_progress": finger_progress, "finger_gate": finger_gate, "thumb_phase": self.steps < self.thumb_phase_steps, "gaps": gaps.copy(), "contacts": contacts.copy(), "opposite_side": opposite_side, "palm_horizontal": palm_horizontal, "penetration_depth": float(np.max(penetration)), "target_tips": self.target_tips.copy(), "target_gaps": self.target_gaps.copy(), "target_qpos": self.target_qpos.copy()}
        return self._observation(), float(reward), terminated, truncated, info


