"""训练拇指位置策略：只优化拇指与手掌高度，不训练四指闭合。"""
from __future__ import annotations
import argparse
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from thumb_position_env import AeroThumbPositionEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('rl/checkpoints/aero_thumb_position_ppo'))
    parser.add_argument('--seed', type=int, default=17)
    args=parser.parse_args()
    check_env(AeroThumbPositionEnv(), warn=True)
    vec=DummyVecEnv([lambda: Monitor(AeroThumbPositionEnv())])
    vec=VecNormalize(vec,norm_obs=True,norm_reward=True,clip_obs=10.0)
    model=PPO('MlpPolicy',vec,verbose=1,seed=args.seed,n_steps=1024,batch_size=256,learning_rate=3e-4,gamma=0.98,gae_lambda=0.95,ent_coef=0.005,tensorboard_log='runs/tensorboard')
    model.learn(total_timesteps=args.timesteps,progress_bar=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);model.save(args.output);vec.save(str(args.output)+'_vecnormalize.pkl');vec.close()
    print(f'已保存拇指定位策略：{args.output}.zip')

if __name__=='__main__': main()
