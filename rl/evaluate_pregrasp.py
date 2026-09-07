"""评估 Pregrasp 策略。"""
from __future__ import annotations
import argparse
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pregrasp_env import AeroPregraspEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_pregrasp_ppo"))
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    vec = DummyVecEnv([lambda: AeroPregraspEnv()])
    stats = pathlib.Path(str(args.model) + "_vecnormalize.pkl")
    if stats.exists():
        vec = VecNormalize.load(str(stats), vec)
        vec.training = False
        vec.norm_reward = False
    model = PPO.load(str(args.model), env=vec, device="cpu")
    success = 0
    errors = []
    for ep in range(args.episodes):
        obs = vec.reset()
        done = [False]
        info = {}
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = vec.step(action)
            info = infos[0]
        success += int(info.get("success", False))
        errors.append(float(info.get("mean_gap_error", float("nan"))))
        print(f"episode {ep+1}: success={info.get('success')} gap_error={errors[-1]:.4f} m penetration={float(info.get('penetration_depth', 0.0)):.4f} m")
    print(f"Pregrasp 成功率：{success}/{args.episodes}，平均间隙误差：{sum(errors)/len(errors):.4f} m")
    vec.close()


if __name__ == "__main__":
    main()
