# 竖直侧向夹持的分阶段强化学习方案

## 设计依据

本项目采用“阶段奖励 + 阶段门控”的课程学习方式。这个结构参考了 [robosuite 的分阶段抓取奖励](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/stack.py)、[UR3_ROS2_PICK_AND_PLACE 的 Reach→Grasp→Lift→Place 流程](https://github.com/darshmenon/UR3_ROS2_PICK_AND_PLACE)，以及 [Bi-DexHands 的 ShadowHand 多指任务](https://github.com/PKU-MARL/DexterousHands)。这些项目共同的做法是：先让末端到达预抓取位，再检查真实接触和夹持，最后才开放抬升奖励。

## 每个阶段

### 0. 几何预校准

先用指尖几何计算预抓取锚点，不让 PPO 同时学习坐标系和抓取动作。球心沿掌面法向留出 25 mm 间隙；拇指与四指的中点在水平面内对准球体。这个间隙避免拇指一开始就穿入球体，也让“球在两侧之间”成为可观察的几何目标。

### 1. Reach（已实现）

[reach_env.py](reach_env.py) 只训练手腕的 `hand_x`、`hand_y` 两个水平滑轨，手掌四元数固定，因此不会通过旋转手腕改变姿态。手指执行器保持复位时的腱长，球体固定在支撑面上方。

奖励由以下部分组成：

- `exp(-水平锚点误差 / 35 mm)`：主要的稠密接近奖励；
- `exp(-XY误差 / 25 mm)`：加快水平定位；
- 掌面法向水平度：保持手掌平面竖直；
- 拇指/四指位于球体相对两侧：预抓取结构奖励；
- 穿透深度和动作变化惩罚：防止碰撞和抖动。

连续 20 步满足水平误差小于 5 mm、两侧关系正确且无明显穿透，即判定 Reach 成功。

### 2. Pregrasp / Shape

把 Reach 的末状态作为初始状态，开放拇指外展和四指屈曲的动作；保持 `hand_x`、`hand_y` 锁定。奖励指尖到球面的安全间隙、拇指和四指的侧向符号，以及四指指腹朝向球心。这里不奖励抬升。

### 3. Contact / Squeeze

只有检测到真实指尖-球体接触后才增加夹紧奖励。对每个指尖记录法向力 `N`、切向力 `T` 和穿透深度，并使用摩擦裕量 `μN-|T|`。成功条件要求拇指和至少三根手指接触、位于相对两侧、穿透小于 1 mm，并持续保持若干步。这样可以避免只靠单侧挤压获得高分。

### 4. Lift / Hold

只有 `stable_grasp` 成立后才开放 `hand_lift`。抬升奖励使用球体高度增量；同时惩罚球体相对手的位移、线速度、角速度、支撑面接触和掉落。手掌姿态奖励贯穿整个阶段，抬升动作只改变整只手的竖直平移，不改变手腕朝向。

## 为什么不直接平均四个对指策略

原有四个对指策略学习的是“拇指和单指相碰”。把它们的动作直接平均会把四根手指拉到拇指同一侧，破坏球体两侧的夹持几何。因此后续只把它们作为手指闭合时序的先验或残差正则，不再把先验动作直接覆盖到手腕和全部手指动作上。

## 运行

```powershell
# 第一阶段短测试
.\\.conda\\python.exe rl/train_reach.py --timesteps 2048 --output rl/checkpoints/aero_reach_smoketest2

# 正式训练第一阶段
.\\.conda\\python.exe rl/train_reach.py --timesteps 120000 --output rl/checkpoints/aero_reach_ppo

# 评估
.\\.conda\\python.exe rl/evaluate_reach.py --model rl/checkpoints/aero_reach_ppo --episodes 10
```

Reach 模型使用单独的 [right_hand_reach.xml](right_hand_reach.xml)，不会覆盖当前已经训练好的旧模型；后续把 Reach 末状态接入多指夹持环境时，需要用同一套扩展后的模型重新训练后续阶段。


## 拇指定位阶段（已实现）

[thumb_position_env.py](thumb_position_env.py) 先通过几何校准把手腕沿世界 Z 方向移动，使拇指当前指尖和球心处于同一高度；随后只开放三个拇指执行器，四指和水平腕部位置全部锁定。目标点为球心外侧的拇指接触方向，回放会显示拇指位置误差、高度误差和侧向误差。

```powershell
.\.conda\python.exe rl\train_thumb_position.py --timesteps 100000 --output rl\checkpoints\aero_thumb_position_ppo
.\.conda\python.exe rl\evaluate_thumb_position.py --model rl\checkpoints\aero_thumb_position_calibrated_v2 --episodes 10
.\.conda\python.exe rl\replay_thumb_position.py --model rl\checkpoints\aero_thumb_position_calibrated_v2
```

一次 20,000 步校准训练达到 9/10 成功，平均拇指位置误差约 9.6 mm；这一步成功后再把拇指策略作为 Pregrasp 的初始化状态。

## Pregrasp 阶段（已实现）

[pregrasp_env.py](pregrasp_env.py) 使用 Reach 阶段同一套扩展手模型。每次复位先根据球心、四指指尖和拇指指尖计算腕部水平目标，然后锁定 `hand_x`、`hand_y` 和 `hand_lift`。策略只输出七个手指/拇指执行器的渐进式控制量，目标是让四指靠近球体 +Y 侧、拇指靠近 -Y 侧，同时保持 5 mm 左右安全间隙。

穿透深度超过 3 mm 会终止回合；连续 25 步满足指尖误差小于 10 mm、两侧关系正确且手掌法向保持水平，才算 Pregrasp 成功。这个阶段仍不计算抬升奖励，也不允许手腕重新寻找球体。

```powershell
.\.conda\python.exe rl\train_pregrasp.py --timesteps 150000 --output rl\checkpoints\aero_pregrasp_ppo
.\.conda\python.exe rl\evaluate_pregrasp.py --model rl\checkpoints\aero_pregrasp_ppo --episodes 10
.\.conda\python.exe rl\replay_pregrasp.py --model rl\checkpoints\aero_pregrasp_ppo
```

## 默认可视化方式

所有阶段的策略回放统一采用三视图进度条窗口。Reach 阶段使用：

```powershell
.\.conda\python.exe rl\replay_reach.py --model rl\checkpoints\aero_reach_ppo_v1
```

后续 Pregrasp、Squeeze、Lift 阶段沿用相同界面约定：斜视、侧视、顶视、可拖动进度条、播放/暂停、逐帧按钮，并显示阶段对应的误差、接触、穿透和成功指标。

