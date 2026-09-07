"""拇指依次触碰食指、中指、无名指和小指的预训练环境。"""

from __future__ import annotations

import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


SCENE = pathlib.Path(__file__).resolve().parent / "opposition_scene.xml"
FINGER_NAMES = ("食指", "中指", "无名指", "小指")
FINGER_BODIES = ("right_index", "right_middle", "right_ring", "right_pinky")


class AeroOppositionEnv(gym.Env):
    """让拇指分别与四根手指建立真实几何接触。"""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 300, target_index: int | None = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.steps = 0
        self.hold_steps = 0
        self.previous_distance = 0.0
        self.required_hold_steps = 50
        self.touch_distance = 0.020
        self.target_index = 0
        self.fixed_target_index = target_index

        # 指尖 site 是任务的几何观测；拇指是最后一个 site。
        self.tip_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")
        ]
        # 接触奖励只认拇指尖端和目标手指尖端，避免指节或掌部擦碰被误判为对指成功。
        self.thumb_tip_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "th_tip")
        self.target_tip_geoms = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("if_tip", "mf_tip", "rf_tip", "pf_tip")
        ]
        self.thumb_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("right_thumb_mcp_link", "right_thumb_proximal_link", "right_thumb_distal_link")
        }
        self.target_body_ids = []
        for prefix in FINGER_BODIES:
            self.target_body_ids.append(
                {
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_f_link"),
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_proximal_link"),
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_middle_link"),
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_distal_link"),
                }
            )
        self.hand_lift_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_lift")
        self.hand_lift_qpos = self.model.jnt_qposadr[self.hand_lift_joint]

        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.previous_action = np.zeros(self.model.nu, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        # 关节状态、执行器状态、四个相对指尖向量和当前目标 one-hot。
        obs_size = self.model.nq + self.model.nv + self.model.nu + 12 + 4
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

    def _observation(self) -> np.ndarray:
        """拼接包含目标指头信息的观测向量。"""
        tips = self.data.site_xpos[self.tip_sites]
        relative = (tips[:4] - tips[4]).reshape(-1)
        target = np.zeros(4, dtype=np.float32)
        target[self.target_index] = 1.0
        return np.concatenate((self.data.qpos, self.data.qvel, self.data.ctrl, relative, target)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        """复位手部，并选择一个需要触碰的手指目标。"""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # 有 seed 时按四根手指循环，便于评估；训练时没有 seed 则随机选目标。
        if self.fixed_target_index is not None:
            self.target_index = int(self.fixed_target_index)
        else:
            self.target_index = int(seed) % 4 if seed is not None else int(self.np_random.integers(0, 4))
        self.data.qpos[self.hand_lift_qpos] = 0.0
        self.data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.previous_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.hold_steps = 0
        tips = self.data.site_xpos[self.tip_sites]
        self.previous_distance = float(np.linalg.norm(tips[self.target_index] - tips[4]))
        return self._observation(), {"target_index": self.target_index, "target_name": FINGER_NAMES[self.target_index]}

    def _target_contacts(self) -> int:
        """统计拇指尖端与当前目标手指尖端的真实碰撞数量。"""
        count = 0
        target_tip_geom = self.target_tip_geoms[self.target_index]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1 == self.thumb_tip_geom and geom2 == target_tip_geom) or (
                geom2 == self.thumb_tip_geom and geom1 == target_tip_geom
            ):
                count += 1
        return count

    def step(self, action):
        """执行对指动作，抬升关节在本任务中始终锁定。"""
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        low, high = self.model.actuator_ctrlrange.T
        # 中指、无名指和小指的固定目标训练只让目标手指和拇指运动，其余三指保持张开，降低动作干扰；食指保留旧策略兼容路径。
        if self.fixed_target_index is not None and self.target_index != 0:
            active_action = np.ones_like(action)
            active_action[self.target_index] = action[self.target_index]
            active_action[4:7] = action[4:7]
            active_action[7] = 0.0
        else:
            active_action = action
        self.last_action = 0.85 * self.last_action + 0.15 * active_action
        self.last_action[7] = 0.0
        self.data.ctrl[:] = low + (self.last_action + 1.0) * 0.5 * (high - low)
        self.data.ctrl[7] = 0.0
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[self.hand_lift_qpos] = 0.0
        self.data.qvel[self.hand_lift_qpos] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        tips = self.data.site_xpos[self.tip_sites]
        distances = np.linalg.norm(tips[:4] - tips[4], axis=1)
        target_distance = float(distances[self.target_index])
        # 计算两个指腹 geom 的实际最短间隙，比 site 中心距离更能反映是否真的贴合。
        fromto = np.zeros(6, dtype=np.float64)
        geom_gap = float(
            max(
                0.0,
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    self.thumb_tip_geom,
                    self.target_tip_geoms[self.target_index],
                    1.0,
                    fromto,
                ),
            )
        )
        contacts = self._target_contacts()
        # 使用有界 tolerance 距离奖励：进入 20 mm 内达到满分，超过 45 mm 逐渐降为 0。
        # 这种有界形式借鉴 MuJoCo Playground，避免远距离状态的指数奖励过度主导训练。
        tolerance_margin = 0.025
        tolerance_bound = 0.020
        site_distance_reward = float(
            np.clip(1.0 - max(0.0, target_distance - tolerance_bound) / tolerance_margin, 0.0, 1.0)
        )
        # 指腹 geom 间隙单独给出更强的近距离梯度，鼓励真正贴合而非只移动 site 中心。
        gap_reward = float(np.clip(1.0 - geom_gap / 0.015, 0.0, 1.0))
        distance_reward = 0.5 * site_distance_reward + 1.5 * gap_reward
        # 额外奖励本步距离缩短，鼓励持续接近而不是停在一个近似位置。
        progress = float(np.clip((self.previous_distance - target_distance) / 0.005, 0.0, 1.0))
        progress_reward = 3.0 * progress
        # 接触奖励只在指腹距离足够近时生效，避免远距离的指节擦碰获得奖励。
        # 拇指和目标指尖各有 1 mm 接触 margin；成功允许合计约 2 mm 的指腹接触间隙。
        touching = bool(geom_gap <= 0.0021 and contacts > 0)
        contact_reward = 10.0 if touching else 0.0
        # 保持奖励随连续接触时间增加，鼓励稳定维持而不是短暂碰撞。
        hold_reward = 0.2 * min(self.hold_steps, self.required_hold_steps) if touching else 0.0
        # 借鉴 Tetheria 示例，惩罚动作变化率而不是动作绝对值，减少不必要抖动。
        action_rate = float(np.mean((self.last_action[:7] - self.previous_action[:7]) ** 2))
        reward = distance_reward + progress_reward + contact_reward + hold_reward - 0.01 * action_rate
        self.hold_steps = self.hold_steps + 1 if touching else 0
        success = self.hold_steps >= self.required_hold_steps
        if success:
            reward += 40.0
        terminated = bool(success)
        truncated = self.steps >= self.max_steps
        info = {
            "target_index": self.target_index,
            "target_name": FINGER_NAMES[self.target_index],
            "target_distance": target_distance,
            "geom_gap": geom_gap,
            "gap_reward": gap_reward,
            "all_distances": distances.copy(),
            "contacts": contacts,
            "touching": touching,
            "hold_steps": self.hold_steps,
            "success": success,
            "distance_reward": distance_reward,
            "progress_reward": progress_reward,
            "contact_reward": contact_reward,
            "hold_reward": hold_reward,
            "action_rate": action_rate,
        }
        self.previous_distance = target_distance
        self.previous_action = self.last_action.copy()
        return self._observation(), float(reward), terminated, truncated, info
