"""Replay a trained PPO policy in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco.viewer
from stable_baselines3 import PPO

from aero_grasp_env import AeroGraspEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a trained Aero Hand PPO policy")
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_grasp_ppo"))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    if args.model.is_absolute():
        model_path = args.model
    elif (pathlib.Path.cwd() / args.model).exists() or (pathlib.Path.cwd() / f"{args.model}.zip").exists():
        model_path = pathlib.Path.cwd() / args.model
    else:
        model_path = pathlib.Path(__file__).resolve().parent / args.model
    env = AeroGraspEnv()
    policy = PPO.load(model_path, env=env)
    observation, _ = env.reset(seed=0)
    episode = 0

    print(f"Loaded policy: {model_path}")
    print("Close the MuJoCo window to stop playback.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episode < args.episodes:
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                episode += 1
                print(f"episode={episode} success={info['success']} lift={info['lift']:.4f} reward={reward:.3f}")
                if episode < args.episodes:
                    observation, _ = env.reset(seed=episode)
            time.sleep(0.005)
    env.close()


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
