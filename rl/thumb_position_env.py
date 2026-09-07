"""拇指单独定位阶段：先把拇指指腹移动到球体侧面的同高度位置。"""
from __future__ import annotations
import pathlib
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE = pathlib.Path(__file__).resolve().parent / "reach_scene.xml"
TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")


class AeroThumbPositionEnv(gym.Env):
    """锁定腕部和四指，只训练拇指到达球体侧面的目标点。"""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 240, object_position=None, hold_steps: int = 25):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.max_steps = int(max_steps)
        self.required_hold_steps = int(hold_steps)
        self.steps = 0
        self.hold_count = 0
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_sphere")
        self.support_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "support_surface")
        self.mount_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tetheria_mount")
        self.palm_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.tip_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "th_tip")
        self.tip_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "th_tip")
        self.slide_qpos = {n: self._joint_qpos(n) for n in ("hand_lift", "hand_x", "hand_y")}
        self.slide_actuators = {n: self._actuator(n + "_A") for n in ("hand_lift", "hand_x", "hand_y")}
        self.thumb_actuators = [self._actuator(n) for n in ("right_thumb_A_cmc_abd", "right_th1_A_tendon", "right_th2_A_tendon")]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # 机械状态、拇指相对目标、球面间隙、掌面方向和时间阶段。
        obs_size = self.model.nq + self.model.nv + 3 + 3 + 1 + 3 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.initial_object_position = np.asarray(object_position if object_position is not None else (0.10, -0.03, -0.03), dtype=np.float64)
        self.object_position = self.initial_object_position.copy()
        self.target_point = np.zeros(3, dtype=np.float64)
        self.target_qpos = np.zeros(3, dtype=np.float64)
        self.last_thumb_action = np.zeros(3, dtype=np.float32)
        self.previous_error = 0.0

    def _joint_qpos(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[jid])

    def _actuator(self, name):
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))

    def _thumb_gap(self):
        return float(mujoco.mj_geomDistance(self.model, self.data, self.tip_geom, self.object_geom, 1.0, np.zeros(6)))

    def _lock_wrist(self):
        # 三个滑轨全部锁定；竖直高度由几何校准预先对齐到球心高度。
        for i, name in enumerate(("hand_lift", "hand_x", "hand_y")):
            self.data.qpos[self.slide_qpos[name]] = self.target_qpos[i]
            self.data.qvel[self.slide_qpos[name]] = 0.0

    def _observation(self):
        tip = self.data.site_xpos[self.tip_site]
        object_pos = self.data.xpos[self.object_body]
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        return np.concatenate((
            self.data.qpos.copy(), self.data.qvel.copy(),
            tip - object_pos, tip - self.target_point,
            np.array([self._thumb_gap()]), palm_axes[:, 1],
            np.array([self.steps / max(1, self.max_steps)]),
        )).astype(np.float32)

    def _set_wrist_target(self):
        # 让手腕先处于预抓取的水平位置，但不改变竖直姿态。
        all_tips = self.data.site_xpos[[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIP_NAMES]]
        anchor = 0.5 * (all_tips[:4].mean(axis=0) + all_tips[4])
        desired_anchor = self.object_position + np.array([0.025, 0.030, 0.0])
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.object_position = self.initial_object_position + np.array([
            self.np_random.uniform(-0.010, 0.010), self.np_random.uniform(-0.012, 0.012), 0.0
        ])
        self.model.body_pos[self.object_body] = self.object_position
        self.model.geom_pos[self.support_geom, 2] = self.object_position[2] - 0.09
        mujoco.mj_forward(self.model, self.data)
        self._set_wrist_target()
        # 几何预校准：直接下移手腕，使当前拇指指尖与球心处于同一世界 Z 高度。
        self.target_qpos[0] = 0.0
        self._lock_wrist()
        mujoco.mj_forward(self.model, self.data)
        thumb_height_delta = self.object_position[2] - self.data.site_xpos[self.tip_site][2]
        self.target_qpos[0] = float(np.clip(thumb_height_delta, -0.14, 0.14))
        # 目标点位于球体拇指侧，法向方向留出约 25 mm 安全距离。
        self.target_point = self.object_position + np.array([0.025, -0.030, 0.0])
        self._lock_wrist()
        mujoco.mj_forward(self.model, self.data)
        low, high = self.model.actuator_ctrlrange.T
        neutral = np.clip(self.data.actuator_length.copy(), low, high)
        self.data.ctrl[:] = neutral
        self.last_thumb_action[:] = [np.clip(2.0 * (neutral[i] - low[i]) / (high[i] - low[i]) - 1.0, -1.0, 1.0) for i in self.thumb_actuators]
        self.steps = 0
        self.hold_count = 0
        tip = self.data.site_xpos[self.tip_site]
        self.previous_error = float(np.linalg.norm(tip - self.target_point))
        return self._observation(), {"object_pos": self.object_position.copy(), "target_point": self.target_point.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        # 用小步残差改变拇指三个执行器，零动作保持当前姿态。
        self.last_thumb_action = np.clip(self.last_thumb_action + 0.15 * action, -1.0, 1.0)
        low, high = self.model.actuator_ctrlrange.T
        ctrl = np.clip(self.data.ctrl.copy(), low, high)
        for i, actuator_id in enumerate(self.thumb_actuators):
            ctrl[actuator_id] = low[actuator_id] + (self.last_thumb_action[i] + 1.0) * 0.5 * (high[actuator_id] - low[actuator_id])
        self.data.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self._lock_wrist()
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        tip = self.data.site_xpos[self.tip_site].copy()
        object_pos = self.data.xpos[self.object_body].copy()
        error_vec = tip - self.target_point
        position_error = float(np.linalg.norm(error_vec))
        height_error = abs(float(tip[2] - object_pos[2]))
        horizontal_error = float(np.linalg.norm((tip - self.target_point)[:2]))
        side_error = abs(float(tip[1] - self.target_point[1]))
        gap = self._thumb_gap()
        penetration = max(0.0, -gap)
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        palm_horizontal = float(np.linalg.norm(palm_axes[:, 1][:2]))
        thumb_side = float(tip[1] - object_pos[1])
        side_ok = thumb_side < -0.012
        progress = self.previous_error - position_error
        self.previous_error = position_error
        reward = (
            4.0 * np.exp(-position_error / 0.035)
            + 6.0 * np.exp(-height_error / 0.012)
            + 3.0 * np.exp(-side_error / 0.018)
            + 1.0 * float(side_ok)
            + 1.0 * palm_horizontal
            + 4.0 * progress
            - 0.02 * float(np.mean(np.square(action)))
            - 120.0 * penetration
        )
        aligned = bool(height_error < 0.006 and side_error < 0.008 and horizontal_error < 0.012 and side_ok and penetration < 0.001)
        self.hold_count = self.hold_count + 1 if aligned else 0
        success = self.hold_count >= self.required_hold_steps
        terminated = bool(success or penetration > 0.003)
        truncated = self.steps >= self.max_steps
        info = {"success": success, "aligned": aligned, "hold_count": self.hold_count, "target_point": self.target_point.copy(), "thumb_position": tip.copy(), "position_error": position_error, "height_error": height_error, "horizontal_error": horizontal_error, "side_error": side_error, "thumb_side": thumb_side, "gap": gap, "penetration_depth": penetration, "palm_horizontal": palm_horizontal, "progress": progress}
        return self._observation(), float(reward), terminated, truncated, info


