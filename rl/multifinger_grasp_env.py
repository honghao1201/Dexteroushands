"""基于对指先验的多指球体静态夹紧强化学习环境。"""

from __future__ import annotations

import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from grasp_prior import OppositionPrior


SCENE = pathlib.Path(__file__).resolve().parent / "grasp_scene.xml"
STATIC_SCENE = pathlib.Path(__file__).resolve().parent / "multifinger_static_scene.xml"
FINGER_NAMES = ("index", "middle", "ring", "pinky")
TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")


class AeroMultiFingerGraspEnv(gym.Env):
    """训练拇指和四根手指同时夹住球体；默认先冻结球体，专注验证夹紧几何。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 300,
        freeze_object: bool = True,
        use_prior: bool = False,
        residual_scale: float = 0.35,
        object_position: np.ndarray | tuple[float, float, float] | None = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(STATIC_SCENE if freeze_object else SCENE))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.freeze_object = freeze_object
        self.residual_scale = float(residual_scale)
        self.steps = 0
        self.hold_steps = 0
        self.required_hold_steps = 30
        self.support_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "support_surface")
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_sphere")
        self.object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.has_object_joint = self.object_joint >= 0
        self.object_qpos = self.model.jnt_qposadr[self.object_joint] if self.has_object_joint else 0
        # 静态夹紧预训练使用较近的球体位置，先验证多指接触几何；hard 场景仍使用 XML 原始位置。
        if object_position is None and freeze_object:
            object_position = (0.10, -0.03, -0.03)
        self.initial_object_position = (
            np.asarray(object_position, dtype=np.float64).copy()
            if object_position is not None
            else self.model.body_pos[self.object_body].copy()
        )
        self.support_z = float(self.initial_object_position[2] - 0.025)
        if freeze_object:
            # 固定场景没有 freejoint，直接修改球体刚体的世界位置以支持 hard 场景复现。
            self.model.body_pos[self.object_body] = self.initial_object_position
        if freeze_object and object_position is not None:
            # 预训练场景中让球体仍然落在支撑面上；只改变球的基准位置，不改手掌姿态。
            self.model.geom_pos[self.support_geom, 2] = self.support_z
        self.tip_sites = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in TIP_NAMES]
        self.tip_geoms = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in TIP_NAMES]
        self.palm_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.hand_lift_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_lift")
        self.hand_lift_qpos = self.model.jnt_qposadr[self.hand_lift_joint]
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        # 关节、速度、控制量，加上物体相对位置/速度、五个指尖相对位置、间隙、接触、力和掌面方向。
        obs_size = 17 + 17 + self.model.nu + 3 + 6 + 15 + 5 + 5 + 10 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.prior = OppositionPrior() if use_prior else None

    def _tip_data(self):
        tips = self.data.site_xpos[self.tip_sites].copy()
        gaps = np.zeros(5, dtype=np.float32)
        contacts = np.zeros(5, dtype=np.float32)
        normal = np.zeros(5, dtype=np.float32)
        tangential = np.zeros(5, dtype=np.float32)
        force = np.zeros(6, dtype=np.float64)
        for i, geom_id in enumerate(self.tip_geoms):
            fromto = np.zeros(6, dtype=np.float64)
            # mj_geomDistance 返回有符号距离：负值表示两个几何体已经发生穿透。
            gaps[i] = float(mujoco.mj_geomDistance(self.model, self.data, geom_id, self.object_geom, 1.0, fromto))
            for contact_id in range(self.data.ncon):
                contact = self.data.contact[contact_id]
                if {int(contact.geom1), int(contact.geom2)} != {geom_id, self.object_geom}:
                    continue
                contacts[i] += 1.0
                mujoco.mj_contactForce(self.model, self.data, contact_id, force)
                normal[i] += max(0.0, float(force[0]))
                tangential[i] += float(np.linalg.norm(force[1:3]))
        # 允许极小数值误差，但明显穿透不再被当作有效接触。
        touching = ((gaps >= -0.0005) & (gaps <= 0.0025) & (contacts > 0)).astype(np.float32)
        return tips, gaps, contacts, touching, normal, tangential

    def _observation(self) -> np.ndarray:
        """构造包含接触和摩擦信息的观测。"""
        tips, gaps, _, touching, normal, tangential = self._tip_data()
        object_pos = self.data.xpos[self.object_body]
        object_vel = self.data.qvel[self.object_qpos : self.object_qpos + 6] if self.has_object_joint else np.zeros(6, dtype=np.float64)
        relative_tips = (tips - object_pos).reshape(-1)
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        palm_normal = palm_axes[:, 1]
        return np.concatenate(
            (
                self.data.qpos[:17],
                self.data.qvel[:17],
                self.data.ctrl,
                object_pos - self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")],
                object_vel,
                relative_tips,
                gaps,
                touching,
                np.clip(normal, 0.0, 10.0),
                np.clip(tangential, 0.0, 10.0),
                palm_normal,
            )
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        """复位手和球体，只对球体水平位置做小范围随机化。"""
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if self.has_object_joint:
            self.data.qpos[self.object_qpos : self.object_qpos + 3] = self.initial_object_position + np.array(
                [self.np_random.uniform(-0.002, 0.002), self.np_random.uniform(-0.006, 0.006), 0.0]
            )
        self.data.qvel[:] = 0.0
        self.data.qpos[self.hand_lift_qpos] = 0.0
        # 归一化动作 -1 对应较低腱长，作为安全的张开初始控制，避免复位后立刻挤进球体。
        self.last_action[:] = -1.0
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.hold_steps = 0
        return self._observation(), {"object_pos": self.data.xpos[self.object_body].copy()}

    def step(self, action):
        """执行一次多指动作，并计算接触、夹紧和保持奖励。"""
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        prior_action = self.prior.action(self.model, self.data) if self.prior is not None else np.zeros_like(action)
        # 先验负责基本闭合，PPO 只学习修正量；静态阶段锁定整手抬升通道。
        blended = np.clip(prior_action + self.residual_scale * action, -1.0, 1.0)
        if self.freeze_object:
            blended[7] = 0.0
        self.last_action = 0.8 * self.last_action + 0.2 * blended
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = low + (self.last_action + 1.0) * 0.5 * (high - low)
        if self.freeze_object:
            self.data.ctrl[7] = 0.0
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        if self.freeze_object:
            # 静态夹紧阶段把球体恢复到原位，保留指尖碰撞对手部产生的反作用力。
            self.data.qpos[self.object_qpos : self.object_qpos + 7] = np.array(
                [*self.initial_object_position, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            self.data.qvel[self.object_qpos : self.object_qpos + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
        if self.freeze_object:
            self.data.qpos[self.hand_lift_qpos] = 0.0
            self.data.qvel[self.hand_lift_qpos] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        tips, gaps, contacts, touching, normal, tangential = self._tip_data()
        palm_axes = self.data.xmat[self.palm_body].reshape(3, 3)
        palm_normal = palm_axes[:, 1]
        palm_horizontal = float(np.linalg.norm(palm_normal[:2]))
        object_pos = self.data.xpos[self.object_body]
        side_axis = palm_axes[:, 2]
        finger_side = float(np.mean((tips[:4] - object_pos) @ side_axis))
        thumb_side = float((tips[4] - object_pos) @ side_axis)
        opposite_side = float(finger_side * thumb_side < 0.0)
        side_score = min(1.0, min(abs(finger_side), abs(thumb_side)) / 0.02) if opposite_side else 0.0
        finger_touch_count = int(np.sum(touching[:4]))
        thumb_touch = float(touching[4])
        contact_score = finger_touch_count / 4.0
        penetration = np.maximum(0.0, -gaps)
        penetration_depth = float(np.max(penetration))
        gap_score = float(np.mean(np.clip(1.0 - np.maximum(0.0, gaps) / 0.025, 0.0, 1.0)))
        # 单独强化拇指接近球体的梯度，避免策略只挤压三根手指而忽略拇指。
        thumb_gap_score = float(np.clip(1.0 - max(0.0, float(gaps[4])) / 0.025, 0.0, 1.0))
        force_target = 0.30
        force_score = float(np.mean(np.clip(normal[:4] / force_target, 0.0, 1.0)))
        friction_margin = np.maximum(0.0, 1.2 * normal[:4] - tangential[:4])
        friction_score = float(np.mean(np.clip(friction_margin / 0.30, 0.0, 1.0)))
        stable_grasp = bool(thumb_touch > 0.5 and finger_touch_count >= 3 and opposite_side and palm_horizontal > 0.86)
        object_lift = float(object_pos[2] - self.initial_object_position[2])
        object_speed = float(np.linalg.norm(self.data.qvel[self.object_qpos : self.object_qpos + 3]))
        angular_speed = float(np.linalg.norm(self.data.qvel[self.object_qpos + 3 : self.object_qpos + 6]))
        lift_low, lift_high = self.model.actuator_ctrlrange[7]
        lift_command = float(np.clip((self.data.ctrl[7] - lift_low) / (lift_high - lift_low), 0.0, 1.0))
        self.hold_steps = self.hold_steps + 1 if stable_grasp and (self.freeze_object or object_lift > 0.04) else 0
        success = self.hold_steps >= self.required_hold_steps and (self.freeze_object or object_lift > 0.06)
        dropped = bool((not self.freeze_object) and object_pos[2] < self.support_z - 0.04)
        support_contacts = sum(
            1
            for contact_id in range(self.data.ncon)
            if {int(self.data.contact[contact_id].geom1), int(self.data.contact[contact_id].geom2)}
            == {self.object_geom, self.support_geom}
        )
        lift_progress = float(np.clip(object_lift / 0.08, 0.0, 1.0))
        reward = (
            1.5 * gap_score
            + 2.0 * contact_score
            + 2.0 * thumb_touch
            + 5.0 * side_score * contact_score
            + 3.0 * thumb_gap_score
            + 2.0 * force_score * contact_score
            + 1.5 * friction_score * contact_score
            + 3.0 * min(1.0, self.hold_steps / self.required_hold_steps)
            + 1.0 * palm_horizontal
            + (8.0 * lift_progress * float(stable_grasp) if not self.freeze_object else 0.0)
            - ((2.0 * object_speed + 0.5 * angular_speed) if not self.freeze_object else 0.0)
            - 0.01 * float(np.mean(np.square(self.last_action - action)))
            - 4.0 * float(not opposite_side) * max(contact_score, 0.25)
            - 80.0 * float(np.sum(penetration))
        )
        if success:
            reward += 25.0
        if dropped:
            reward -= 20.0
        penetration_violation = bool(penetration_depth > 0.003)
        if penetration_violation:
            reward -= 10.0
        terminated = bool(success or dropped or penetration_violation)
        truncated = self.steps >= self.max_steps
        info = {
            "success": success,
            "stable_grasp": stable_grasp,
            "hold_steps": self.hold_steps,
            "finger_touch_count": finger_touch_count,
            "thumb_touch": thumb_touch,
            "touching": touching.copy(),
            "gaps": gaps.copy(),
            "normal_force": normal.copy(),
            "tangential_force": tangential.copy(),
            "friction_margin": friction_margin.copy(),
            "contact_score": contact_score,
            "thumb_gap_score": thumb_gap_score,
            "force_score": force_score,
            "friction_score": friction_score,
            "opposite_side": bool(opposite_side),
            "side_score": side_score,
            "palm_normal_horizontal": palm_horizontal,
            "object_lift": object_lift,
            "object_speed": object_speed,
            "angular_speed": angular_speed,
            "lift_command": lift_command,
            "lift_progress": lift_progress,
            "support_contacts": support_contacts,
            "dropped": dropped,
            "penetration": penetration.copy(),
            "penetration_depth": penetration_depth,
            "penetration_violation": penetration_violation,
        }
        return self._observation(), float(reward), terminated, truncated, info
