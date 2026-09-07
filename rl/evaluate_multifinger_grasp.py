"""评估多指静态夹紧策略并输出每回合接触和摩擦指标。"""

from __future__ import annotations

import argparse
import pathlib

from stable_baselines3 import PPO

from multifinger_grasp_env import AeroMultiFingerGraspEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_multifinger_static_ppo"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--use-prior", action="store_true")
    parser.add_argument("--hard-scene", action="store_true")
    args = parser.parse_args()
    position = (0.127, -0.015, -0.064) if args.hard_scene or args.dynamic else (0.10, -0.03, -0.03)
    env = AeroMultiFingerGraspEnv(
        freeze_object=not args.dynamic,
        use_prior=args.use_prior,
        object_position=position,
    )
    model = PPO.load(args.model, env=env, device="cpu")
    successes = 0
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=episode)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
        successes += int(info.get("success", False))
        print(
            f"episode={episode + 1} success={info.get('success')} "
            f"finger_contacts={info.get('finger_touch_count')} "
            f"thumb={info.get('thumb_touch', 0):.0f} "
            f"hold={info.get('hold_steps')} "
            f"min_gap={min(info.get('gaps', [0])):.4f} "
            f"min_friction_margin={min(info.get('friction_margin', [0])):.4f}"
        )
    print(f"success_rate={successes}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
