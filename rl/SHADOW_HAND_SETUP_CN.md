# Shadow Hand 接入说明

当前项目保留原来的 Aero Hand 文件，同时新增 Shadow Hand 版本。Shadow Hand 的原生模型有 24 个手部广义自由度，本项目在根部再加入 `hand_lift`、`hand_x`、`hand_y` 和 `hand_yaw` 四个接近自由度，因此 PoseApproach 阶段使用 28 个广义自由度。

接近阶段只控制四个根部自由度，Shadow Hand 的 24 个手指和腕部关节保持张开初始构型。目标位姿由球心、张开手的预抓取锚点和掌面法向解析计算，PPO 只学习参考位姿附近的残差，避免策略把手掌翻转到错误方向。

训练：

```powershell
.\.conda\python.exe rl\train_shadow_pose_approach.py --timesteps 60000 --output rl\checkpoints\shadow_pose_approach_ppo_v1
```

评估：

```powershell
.\.conda\python.exe rl\evaluate_shadow_pose_approach.py --model rl\checkpoints\shadow_pose_approach_ppo_v1 --episodes 10
```

三视图进度回放：

```powershell
.\.conda\python.exe rl\replay_shadow_pose_approach.py --model rl\checkpoints\shadow_pose_approach_ppo_v1
```

模型文件位于 `models/shadow_hand`，其中 `scene_pose.xml` 是带球体和支撑面的训练场景，`right_hand_pose.xml` 是加入根部接近自由度后的右手模型。Aero Hand 的旧 checkpoint 与 Shadow Hand 的 qpos、actuator 数量不同，不能直接混用。
