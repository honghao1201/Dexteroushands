# Shadow Hand 顶部接近阶段

这一阶段只训练整只手移动到球体上方的预抓取位姿，不闭合手指，也不抬升球体。手掌法向目标为世界坐标 `-Z`，锚点目标为球心上方 `0.10 m`。根部新增 `hand_pitch` 自由度，使 Shadow Hand 可以从原来的竖直掌面旋转为掌面朝下。

训练：

```powershell
.\.conda\python.exe rl\train_shadow_topdown_approach.py --timesteps 80000 --output rl\checkpoints\shadow_topdown_approach_ppo_v1
```

评估：

```powershell
.\.conda\python.exe rl\evaluate_shadow_topdown_approach.py --model rl\checkpoints\shadow_topdown_approach_ppo_v1 --episodes 10
```

三视图进度条回放：

```powershell
.\.conda\python.exe rl\replay_shadow_topdown_approach.py --model rl\checkpoints\shadow_topdown_approach_ppo_v1
```

奖励由以下部分组成：

- 锚点到目标位置的指数距离奖励；
- 掌面法向朝向 `-Z` 的姿态奖励；
- 锚点距离和姿态误差的逐步进展奖励；
- 接近目标间隙的安全奖励；
- 穿透、动作幅度和过大间隙惩罚。

成功条件为连续保持 15 步：中心误差小于 8 mm、掌面角度误差小于 0.10 rad、掌面朝下程度大于 0.98，且手部与球体没有穿透。
