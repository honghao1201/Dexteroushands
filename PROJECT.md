# Dexteroushands 项目记录

## 当前目标
在 MuJoCo 中控制 TetherIA Aero Hand：掌面保持竖直，拇指位于球体一侧，其余四指位于另一侧；通过相向闭合建立夹紧力，再沿竖直方向抬升并稳定保持球体。

## 当前状态

- 项目目录：`D:\Projects\Dexteroushands`
- 模型：`models/tetheria_aero_hand_open/`
- Python 环境：`.conda\python.exe`
- 仿真入口：`simulate.py`
- RL 环境：`rl/aero_grasp_env.py`
- 训练入口：`rl/train_ppo.py`
- 回放入口：`rl/evaluate_policy.py`
- 最新模型：`rl/checkpoints/aero_grasp_ppo.zip`
- Git：本地 `main` 分支；默认不自动 push

## 训练验证

最新 PPO 训练完成 100,352 个实际步数。使用 10 个不同随机种子做确定性回放，10/10 回合达到成功条件；完成时间 34–41 个控制步，球体抬升约 0.29 m，整手抬升约 0.24 m，水平速度约 0.003–0.007 m/s，角速度约 0.05–0.09 rad/s，并持续保持 20 步。

这代表当前基线已经学会设定的动作链路，但仍建议在增加球体质量、摩擦和初始位置随机化后继续训练。
