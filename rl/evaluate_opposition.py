"""评估拇指分别触碰四根手指的 PPO 策略。"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco.viewer
from stable_baselines3 import PPO

from opposition_env import AeroOppositionEnv, FINGER_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_opposition_ppo"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--target", type=int, choices=range(4), default=None, help="固定目标手指")
    parser.add_argument(
        "--auto-close",
        action="store_true",
        help="完成指定回合后自动关闭窗口；默认保持窗口打开",
    )
    args = parser.parse_args()

    # 兼容从项目根目录或 rl 目录传入模型路径。
    if args.model.is_absolute():
        model_path = args.model
    elif (pathlib.Path.cwd() / args.model).exists() or (pathlib.Path.cwd() / f"{args.model}.zip").exists():
        model_path = pathlib.Path.cwd() / args.model
    else:
        model_path = pathlib.Path(__file__).resolve().parent / args.model

    # 使用和训练完全相同的无球对指环境加载 PPO 策略。
    env = AeroOppositionEnv(target_index=args.target)
    policy = PPO.load(model_path, env=env)
    successes = 0
    per_finger = [0, 0, 0, 0]

    print(f"Loaded policy: {model_path}")
    print("回放完成后 MuJoCo 窗口会保持打开，请手动关闭窗口退出。")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # 采用斜视角观察拇指和目标手指的接近、接触和保持过程。
        viewer.cam.azimuth = 270
        viewer.cam.elevation = 0
        viewer.cam.distance = 0.35
        viewer.cam.lookat[:] = [0.17, -0.02, -0.01]

        episode = 0
        observation, _ = env.reset(seed=0)
        while viewer.is_running():
            # 达到回合数后停止物理推进，只保留最后画面供检查。
            if episode >= args.episodes:
                viewer.sync()
                if args.auto_close:
                    break
                time.sleep(0.02)
                continue

            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                episode += 1
                successes += int(info["success"])
                per_finger[info["target_index"]] += int(info["success"])
                print(
                    f"episode={episode} target={info['target_name']} success={info['success']} "
                    f"distance={info['target_distance']:.4f} contacts={info['contacts']} reward={reward:.3f}"
                )
                if episode < args.episodes:
                    observation, _ = env.reset(seed=episode)
                elif args.auto_close:
                    break
            time.sleep(0.005)

    print(f"success_rate={successes}/{args.episodes}")
    print("per_finger=" + ", ".join(f"{FINGER_NAMES[i]}:{per_finger[i]}" for i in range(4)))
    env.close()


if __name__ == "__main__":
    # 从项目根目录执行脚本时，确保 rl 目录可导入环境模块。
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
