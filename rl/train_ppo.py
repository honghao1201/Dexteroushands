"""Train a first PPO policy for tendon-space object pickup."""

from __future__ import annotations

import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from aero_grasp_env import AeroGraspEnv


def main() -> None:
    env = AeroGraspEnv()
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/tensorboard")
    model.learn(total_timesteps=1_000_000)
    output = pathlib.Path(__file__).resolve().parent / "checkpoints" / "aero_grasp_ppo"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    env.close()


if __name__ == "__main__":
    main()
