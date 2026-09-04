"""用于竖直球体夹持、抬升和稳定保持的 Gymnasium 环境。"""

from __future__ import annotations

import pathlib  # 计算场景 XML 的绝对路径。

import gymnasium as gym  # 强化学习环境的标准接口。
import mujoco  # MuJoCo 物理仿真接口。
import numpy as np  # 数值计算和观测向量处理。
from gymnasium import spaces  # 定义动作空间和观测空间。


# 项目根目录和当前任务使用的球体抓取场景。
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE = pathlib.Path(__file__).resolve().parent / "grasp_scene.xml"


class AeroGraspEnv(gym.Env):
    """让 PPO 学习拇指/四指对向夹持球体并抬起整只手。"""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500):
        """加载 MuJoCo 模型，并建立强化学习所需的空间和索引。"""
        super().__init__()
        # 加载场景模型和与模型配套的运行时数据。
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        # 回合长度与当前控制步计数器。
        self.max_steps = max_steps
        self.steps = 0

        # 缓存物体、抬升关节、抓取标记点和五个指尖的 MuJoCo 索引。
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_sphere")
        self.object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_qpos = self.model.jnt_qposadr[self.object_joint]
        self.hand_lift_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_lift")
        self.hand_lift_qpos = self.model.jnt_qposadr[self.hand_lift_joint]
        self.grasp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.tip_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")
        ]
        # 支撑面高度，以及成功前必须连续满足稳定条件的控制步数。
        self.support_z = -0.09
        self.required_hold_steps = 20
        self.hold_steps = 0

        # 动作平滑缓存；动作空间的每一维都归一化到 [-1, 1]。
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        # 观测由 qpos、qvel、执行器控制量、球体位置和球体速度拼接而成。
        obs_size = self.model.nq + self.model.nv + self.model.nu + 6 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

    def _observation(self) -> np.ndarray:
        """读取当前状态并拼成固定长度的观测向量。"""
        # 自由球体的 qvel 包含 3 个线速度和 3 个角速度分量。
        object_pos = self.data.xpos[self.object_body]
        object_vel = self.data.qvel[self.object_qpos : self.object_qpos + 6]
        return np.concatenate((self.data.qpos, self.data.qvel, self.data.ctrl, object_pos, object_vel)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        """重置手部姿态和球体位置，返回初始观测。"""
        super().reset(seed=seed)
        # 恢复 XML 中的初始关键帧，再对球体做小范围横向随机化。
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[self.object_qpos : self.object_qpos + 3] = np.array(
            [0.13 + self.np_random.uniform(-0.008, 0.008), -0.015 + self.np_random.uniform(-0.008, 0.008), -0.064]
        )
        # 清零所有速度和动作平滑状态，避免上一个回合的惯性影响新回合。
        self.data.qvel[:] = 0
        self.last_action[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.hold_steps = 0
        # 返回观测，以及便于调试的初始球体位置。
        return self._observation(), {"object_pos": self.data.xpos[self.object_body].copy()}

    def step(self, action):
        """执行一个控制步，推进物理仿真并计算奖励。"""
        # 限制策略输出范围，避免超出执行器控制范围。
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        low, high = self.model.actuator_ctrlrange.T

        # 对腱绳目标做低通平滑，减少不现实的瞬时拉力冲击。
        self.last_action = 0.85 * self.last_action + 0.15 * action

        # 整手抬升通道是单向的：负值和 0 表示不抬升，正值表示向上抬升。
        # 这样未训练策略的零均值输出不会立刻把手和球体弹起。
        target_action = self.last_action.copy()
        target_action[7] = max(0.0, float(target_action[7]))
        self.data.ctrl[:] = low + (target_action + 1) * 0.5 * (high - low)
        self.data.ctrl[7] = low[7] + target_action[7] * (high[7] - low[7])

        # 每个控制步包含 5 个 MuJoCo 物理积分步。
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        # 读取球体、指尖、夹紧程度和抬升执行器状态。
        object_pos = self.data.xpos[self.object_body]
        hand_pos = np.mean(self.data.site_xpos[self.tip_sites], axis=0)
        distance = float(np.linalg.norm(object_pos - hand_pos))
        lift = float(object_pos[2] - self.support_z)
        hand_lift = float(self.data.qpos[self.hand_lift_qpos])
        closure = float(np.mean((self.data.ctrl[:4] - low[:4]) / (high[:4] - low[:4])))
        thumb_closure = float(np.mean((self.data.ctrl[5:7] - low[5:7]) / (high[5:7] - low[5:7])))
        lift_command = float(np.clip((self.data.ctrl[7] - low[7]) / (high[7] - low[7]), 0.0, 1.0))
        thumb_contacts, finger_contacts, normal_force = self._object_contacts()
        opposing_contact = min(1.0, thumb_contacts) * min(1.0, finger_contacts / 2.0)
        horizontal_speed = float(np.linalg.norm(self.data.qvel[self.object_qpos : self.object_qpos + 2]))
        vertical_speed = float(abs(self.data.qvel[self.object_qpos + 2]))
        angular_speed = float(np.linalg.norm(self.data.qvel[self.object_qpos + 3 : self.object_qpos + 6]))
        # 成功需要球体离开支撑面、整手抬高、两侧接触，并持续低速稳定保持。
        lifted = lift > 0.065
        stable = bool(
            lifted
            and hand_lift > 0.035
            and lift_command > 0.25
            and opposing_contact > 0.5
            and normal_force > 0.05
            and horizontal_speed < 0.08
            and vertical_speed < 0.12
            and angular_speed < 2.0
        )
        # 只要有一个稳定条件失效，就重新累计保持时间。
        self.hold_steps = self.hold_steps + 1 if stable else 0
        success = self.hold_steps >= self.required_hold_steps
        stable_hold = min(1.0, self.hold_steps / self.required_hold_steps)
        lift_progress = min(1.0, max(0.0, hand_lift) / 0.12)

        # 奖励由接近球体、手指闭合、对向接触、抬升进度、夹紧力和低滑移组成。
        lift_reward = min(max(0.0, lift), 0.08)
        force_reward = min(normal_force, 100.0)
        reward = (
            1.5 * max(0.0, 1.0 - distance / 0.07)
            + 0.8 * closure
            + 0.8 * thumb_closure
            + 4.0 * opposing_contact
            + 12.0 * lift_reward * opposing_contact * lift_command
            + 4.0 * opposing_contact * lift_progress
            + 1.0 * opposing_contact * lift_command
            + 4.0 * stable_hold * opposing_contact
            + 0.03 * force_reward * opposing_contact
            - 0.5 * horizontal_speed
        )
        # 成功或明确掉落/远离后结束回合；否则让策略继续学习。
        terminated = bool(success)
        dropped = bool(object_pos[2] < self.support_z - 0.07 or distance > 0.35)
        if success:
            reward += 15.0
        if dropped:
            reward -= 10.0
        terminated = terminated or dropped
        truncated = self.steps >= self.max_steps
        # 返回调试指标，评估脚本会用其中的 success 和 lift 字段打印结果。
        info = {
            "distance": distance,
            "lift": lift,
            "hand_lift": hand_lift,
            "thumb_contacts": thumb_contacts,
            "finger_contacts": finger_contacts,
            "opposing_contact": opposing_contact,
            "normal_force": normal_force,
            "lift_command": lift_command,
            "horizontal_speed": horizontal_speed,
            "vertical_speed": vertical_speed,
            "angular_speed": angular_speed,
            "hold_steps": self.hold_steps,
            "stable": stable,
            "success": success,
        }
        return self._observation(), reward, terminated, truncated, info

    def _object_contacts(self) -> tuple[int, int, float]:
        """统计球体与拇指/其余手指的接触数量和法向力。"""
        # 遍历当前所有接触，只保留涉及目标球体的接触。
        thumb_contacts = 0
        finger_contacts = 0
        normal_force = 0.0
        force = np.zeros(6, dtype=np.float64)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 != self.object_geom and contact.geom2 != self.object_geom:
                continue
            # 找到球体以外的几何体，并通过几何体/刚体名称判断属于哪根手指。
            other_geom = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
            other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(other_geom)) or ""
            other_body = int(self.model.geom_bodyid[other_geom])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other_body) or ""
            contact_name = f"{other_name} {body_name}".lower()
            # MuJoCo 接触力的第 0 个分量是法向力，累加用于夹紧力奖励。
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal_force += max(0.0, float(force[0]))
            if "thumb" in contact_name:
                thumb_contacts += 1
            elif any(finger in contact_name for finger in ("index", "middle", "ring", "pinky")):
                finger_contacts += 1
        return thumb_contacts, finger_contacts, normal_force
