"""评估 PoseApproach 策略的位姿对齐效果。"""
from __future__ import annotations

import argparse
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pose_approach_env import AeroPoseApproachEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_pose_approach_ppo_v1"))
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    env = DummyVecEnv([lambda: AeroPoseApproachEnv()])
    stats = pathlib.Path(str(args.model) + "_vecnormalize.pkl")
    if stats.exists():
        env = VecNormalize.load(str(stats), env)
        env.training = False
        env.norm_reward = False
    model = PPO.load(str(args.model), env=env, device="cpu")

    successes = 0
    anchor_errors = []
    angle_errors = []
    for episode in range(args.episodes):
        obs = env.reset()
        done = [False]
        last_info = {}
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            last_info = infos[0]
        successes += int(last_info.get("success", False))
        anchor_errors.append(float(last_info.get("anchor_error", float("nan"))))
        angle_errors.append(float(last_info.get("angle_error", float("nan"))))
        print(
            f"episode {episode + 1}: success={last_info.get('success')} "
            f"anchor_error={anchor_errors[-1]:.4f} m "
            f"angle_error={angle_errors[-1] * 180.0 / 3.1415926:.2f} deg"
        )
    print(
        f"PoseApproach 成功率：{successes}/{args.episodes}，"
        f"平均中心误差：{sum(anchor_errors) / len(anchor_errors):.4f} m，"
        f"平均方向误差：{sum(angle_errors) / len(angle_errors) * 180.0 / 3.1415926:.2f} deg"
    )
    env.close()


if __name__ == "__main__":
    main()
