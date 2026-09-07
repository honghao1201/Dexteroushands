"""评估拇指同高度定位策略。"""
from __future__ import annotations
import argparse
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from thumb_position_env import AeroThumbPositionEnv

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=pathlib.Path,default=pathlib.Path('rl/checkpoints/aero_thumb_position_ppo'));parser.add_argument('--episodes',type=int,default=10);args=parser.parse_args()
    vec=DummyVecEnv([lambda:AeroThumbPositionEnv()]);stats=pathlib.Path(str(args.model)+'_vecnormalize.pkl')
    if stats.exists(): vec=VecNormalize.load(str(stats),vec);vec.training=False;vec.norm_reward=False
    model=PPO.load(str(args.model),env=vec,device='cpu');success=0;errors=[]
    for ep in range(args.episodes):
        obs=vec.reset();done=[False];info={}
        while not done[0]: obs,_,done,infos=vec.step(model.predict(obs,deterministic=True)[0]);info=infos[0]
        success+=int(info.get('success',False));errors.append(float(info.get('position_error',float('nan'))))
        print(f"episode {ep+1}: success={info.get('success')} position_error={errors[-1]:.4f} m height_error={float(info.get('height_error',float('nan'))):.4f} m side_error={float(info.get('side_error',float('nan'))):.4f} m")
    print(f"拇指定位成功率：{success}/{args.episodes}，平均位置误差：{sum(errors)/len(errors):.4f} m");vec.close()
if __name__=='__main__':main()
