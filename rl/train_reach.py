"""训练第一阶段 Reach 策略：先把手腕移动到球体两侧的预抓取位。"""
from __future__ import annotations
import argparse
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from reach_env import AeroReachEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_reach_ppo"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # 先检查 Gymnasium 接口，再用 VecNormalize 稳定不同距离尺度的奖励。
    check_env(AeroReachEnv(), warn=True)
    vec = DummyVecEnv([lambda: Monitor(AeroReachEnv())])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", vec, verbose=1, seed=args.seed,
        n_steps=1024, batch_size=256, learning_rate=3e-4,
        gamma=0.98, gae_lambda=0.95, ent_coef=0.005,
        tensorboard_log="runs/tensorboard",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    vec.save(str(args.output) + "_vecnormalize.pkl")
    vec.close()
    print(f"已保存 Reach 策略：{args.output}.zip")


if __name__ == "__main__":
    main()

