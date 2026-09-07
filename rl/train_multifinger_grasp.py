"""训练基于对指先验的多指静态夹紧 PPO 策略。"""

from __future__ import annotations

import argparse
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from multifinger_grasp_env import AeroMultiFingerGraspEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--use-prior", action="store_true", help="加载已有对指策略作为动作先验；默认关闭，避免先验把四指拉到拇指同侧")
    parser.add_argument("--hard-scene", action="store_true", help="使用 grasp_scene.xml 中原始球体位置")
    parser.add_argument("--dynamic", action="store_true", help="不冻结球体，直接训练可移动物体")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_multifinger_static_ppo"))
    args = parser.parse_args()

    position = (0.127, -0.015, -0.064) if args.hard_scene or args.dynamic else (0.10, -0.03, -0.03)
    env = AeroMultiFingerGraspEnv(
        freeze_object=not args.dynamic,
        use_prior=args.use_prior,
        object_position=position,
    )
    check_env(env, warn=True)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="runs/tensorboard",
    )
    model.learn(total_timesteps=args.timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    env.close()
    print(f"已保存模型：{args.output}.zip")


if __name__ == "__main__":
    main()
