"""评估第一阶段 Reach 策略，输出接近误差和预抓取成功率。"""
from __future__ import annotations
import argparse
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from reach_env import AeroReachEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_reach_ppo"))
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    env = DummyVecEnv([lambda: AeroReachEnv()])
    stats = pathlib.Path(str(args.model) + "_vecnormalize.pkl")
    if stats.exists():
        env = VecNormalize.load(str(stats), env)
        env.training = False
        env.norm_reward = False
    model = PPO.load(str(args.model), env=env)
    successes = 0
    errors = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = [False]
        last_info = {}
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            last_info = infos[0]
        successes += int(last_info.get("success", False))
        errors.append(last_info.get("anchor_error", float("nan")))
        print(f"episode {ep + 1}: success={last_info.get('success')} anchor_error={last_info.get('anchor_error', float('nan')):.4f} m")
    print(f"Reach 成功率：{successes}/{args.episodes}，平均末端误差：{sum(errors)/len(errors):.4f} m")
    env.close()


if __name__ == "__main__":
    main()
