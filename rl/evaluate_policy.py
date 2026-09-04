"""在 MuJoCo 查看器中回放已经训练好的 PPO 策略。"""

from __future__ import annotations

import argparse  # 解析模型路径和回放回合数。
import pathlib  # 处理模型文件路径。
import sys  # 把 rl 目录加入模块搜索路径。
import time  # 控制回放速度。

import mujoco.viewer  # MuJoCo 可视化窗口。
from stable_baselines3 import PPO  # 加载训练好的 PPO 模型。

from aero_grasp_env import AeroGraspEnv  # 与训练时相同的环境。


def main() -> None:
    """加载策略，在窗口中回放指定数量的抓取回合。"""
    # 支持从项目根目录或 rl 目录传入模型路径。
    parser = argparse.ArgumentParser(description="Visualize a trained Aero Hand PPO policy")
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_grasp_ppo"))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    # 兼容绝对路径、当前目录相对路径和 rl 目录相对路径。
    if args.model.is_absolute():
        model_path = args.model
    elif (pathlib.Path.cwd() / args.model).exists() or (pathlib.Path.cwd() / f"{args.model}.zip").exists():
        model_path = pathlib.Path.cwd() / args.model
    else:
        model_path = pathlib.Path(__file__).resolve().parent / args.model
    # 使用和训练完全相同的环境结构加载 PPO 策略。
    env = AeroGraspEnv()
    policy = PPO.load(model_path, env=env)
    observation, _ = env.reset(seed=0)
    episode = 0

    # 输出加载信息，并启动 MuJoCo 被动查看器。
    print(f"Loaded policy: {model_path}")
    print("Close the MuJoCo window to stop playback.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episode < args.episodes:
            # 策略根据当前观测生成 8 维腱绳、拇指和整手抬升动作。
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                # 每回合结束时输出是否成功以及最终抬升高度。
                episode += 1
                print(f"episode={episode} success={info['success']} lift={info['lift']:.4f} reward={reward:.3f}")
                if episode < args.episodes:
                    # 用不同种子重置球体的横向初始位置。
                    observation, _ = env.reset(seed=episode)
            # 给查看器留出刷新时间，避免回放速度过快。
            time.sleep(0.005)
    # 关闭环境并释放资源。
    env.close()


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
