"""将已经训练好的四个对指策略组合成多指抓取的先验动作。"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np
from stable_baselines3 import PPO

from opposition_env import AeroOppositionEnv


class OppositionPrior:
    """把四个单目标对指策略映射为一个共同的八维动作。"""

    def __init__(self, checkpoint_dir: pathlib.Path | None = None):
        root = pathlib.Path(__file__).resolve().parent
        checkpoint_dir = checkpoint_dir or root / "checkpoints"
        self.models = []
        # 对指环境与抓取环境的手部前 17 个关节完全一致，因此可以直接复用策略。
        for finger in range(4):
            path = checkpoint_dir / f"aero_opposition_ppo_finger{finger}.zip"
            if not path.exists():
                raise FileNotFoundError(f"找不到对指先验模型：{path}")
            prior_env = AeroOppositionEnv(target_index=finger)
            self.models.append(PPO.load(path, env=prior_env, device="cpu"))

        self.tip_names = ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")

    def _observation(self, model: mujoco.MjModel, data: mujoco.MjData, target: int) -> np.ndarray:
        """从抓取场景提取对指策略所需的无球观测。"""
        site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in self.tip_names]
        tips = data.site_xpos[site_ids]
        relative = (tips[:4] - tips[4]).reshape(-1)
        target_one_hot = np.zeros(4, dtype=np.float32)
        target_one_hot[target] = 1.0
        return np.concatenate(
            (data.qpos[:17], data.qvel[:17], data.ctrl, relative, target_one_hot)
        ).astype(np.float32)

    def action(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """逐根读取对指动作，并平均拇指通道。"""
        result = np.zeros(model.nu, dtype=np.float32)
        thumb_actions = []
        for finger, policy in enumerate(self.models):
            obs = self._observation(model, data, finger)
            action, _ = policy.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            result[finger] = action[finger]
            thumb_actions.append(action[4:7])
        result[4:7] = np.mean(np.stack(thumb_actions), axis=0)
        result[7] = 0.0
        return np.clip(result, -1.0, 1.0)
