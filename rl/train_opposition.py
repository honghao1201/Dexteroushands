"""使用 PPO 训练拇指与四根手指逐一对指。"""

from __future__ import annotations

import argparse
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from opposition_env import AeroOppositionEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--target", type=int, choices=range(4), default=None, help="固定目标：0 食指，1 中指，2 无名指，3 小指")
    args = parser.parse_args()
    env = AeroOppositionEnv(target_index=args.target)
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/tensorboard")
    model.learn(total_timesteps=args.timesteps)
    suffix = "" if args.target is None else f"_finger{args.target}"
    output = pathlib.Path(__file__).resolve().parent / "checkpoints" / f"aero_opposition_ppo{suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    env.close()
    print(f"Saved: {output}.zip")


if __name__ == "__main__":
    main()
