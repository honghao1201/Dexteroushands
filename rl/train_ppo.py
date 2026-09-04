"""使用 PPO 训练腱绳驱动灵巧手抓取并竖直抬升球体。"""

from __future__ import annotations

import pathlib  # 生成模型输出路径。
import argparse  # 读取训练步数等命令行选项。

from stable_baselines3 import PPO  # Stable-Baselines3 的 PPO 算法。
from stable_baselines3.common.env_checker import check_env  # 检查 Gymnasium 环境接口。

from aero_grasp_env import AeroGraspEnv  # 当前项目的 MuJoCo 抓取环境。


def main() -> None:
    """创建环境、检查接口、训练策略并保存模型。"""
    # 允许用户通过 --timesteps 调整训练量，默认训练一百万步。
    parser = argparse.ArgumentParser(description="Train PPO for Aero Hand tendon-space grasping")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    # 创建环境并执行 SB3 的标准接口检查。
    env = AeroGraspEnv()
    check_env(env, warn=True)

    # 创建多层感知机策略，并把 TensorBoard 日志写入 runs/tensorboard。
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/tensorboard")
    # 开始与 MuJoCo 环境交互，PPO 会根据奖励更新策略网络。
    model.learn(total_timesteps=args.timesteps)

    # 将训练后的策略保存为不带扩展名的路径，SB3 会生成 .zip 文件。
    output = pathlib.Path(__file__).resolve().parent / "checkpoints" / "aero_grasp_ppo"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    # 释放 MuJoCo 环境资源。
    env.close()


if __name__ == "__main__":
    main()
