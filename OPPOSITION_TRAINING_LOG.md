# 拇指对指预训练日志

## 1. 实验目的

本阶段不训练拿球，而是先训练四个基础动作：拇指分别与食指、中指、无名指和小指形成相对的指腹接触，并在接触后保持稳定。只有四个动作都通过后，才进入球体侧向夹持训练。

实验日期：2026-09-05  
项目目录：`D:\Projects\Dexteroushands`  
仿真器：MuJoCo  
强化学习算法：Stable-Baselines3 PPO  
训练解释器：`.\.conda\python.exe`

## 2. 环境设计

训练场景为 `rl/opposition_scene.xml`，不加入球体，不进行抬升，只保留灵巧手本体。整手抬升关节 `hand_lift` 在环境中锁定为 0，因此策略只能学习手指和拇指的变形。

每个 episode 只选择一根目标手指。通过 `--target` 固定训练目标：

| target | 目标 |
| ---: | --- |
| 0 | 食指 |
| 1 | 中指 |
| 2 | 无名指 |
| 3 | 小指 |

动作空间为 8 维：四根手指腱绳、拇指外展、拇指两路腱绳和整手抬升。对固定目标训练时，中指、无名指和小指只激活目标手指与拇指，其余手指保持张开，以减少动作互相干扰；食指保留兼容旧检查点的控制路径。

## 3. Reward 设置

当前 reward 在 `rl/opposition_env.py` 中实现，主要分为以下几项。

### 3.1 有界 site 距离奖励

site 中心距离使用有界容差，而不是无限制的负距离：

```text
site_distance_reward = clip(1 - max(0, distance - 0.020) / 0.025, 0, 1)
```

距离小于 20 mm 时达到上限，超过约 45 mm 时降为 0。这样可以避免远距离状态的数值主导训练。

### 3.2 指腹 geom 间隙奖励

使用 MuJoCo 的 `mj_geomDistance` 计算拇指尖端碰撞盒与目标手指尖端碰撞盒的实际最短间隙：

```text
gap_reward = clip(1 - geom_gap / 0.015, 0, 1)
distance_reward = 0.5 * site_distance_reward + 1.5 * gap_reward
```

geom 间隙的权重高于 site 中心距离，目的是让策略直接靠近指腹碰撞区域，而不是只让两个 site 的中心位置接近。

### 3.3 距离缩短进展奖励

只奖励本步有效缩短，不奖励远离目标：

```text
progress_reward = 3.0 * clip((previous_distance - current_distance) / 0.005, 0, 1)
```

这样可以降低来回抖动和利用负进展奖励的情况。

### 3.4 接触和保持奖励

拇指和四个指尖碰撞盒均设置 1 mm 的 MuJoCo contact margin。成功接触条件为：

```text
contacts > 0 and geom_gap <= 0.0021 m
```

对应奖励为：

```text
contact_reward = 10.0
hold_reward = 0.2 * min(连续接触步数, 50)
```

接触奖励只认拇指尖端 geom 和目标手指尖端 geom，指节或掌部擦碰不会被算作有效对指。

### 3.5 动作变化率惩罚

借鉴腱绳手任务的控制方式，惩罚相邻动作变化率：

```text
action_rate_penalty = 0.01 * mean((last_action - previous_action)^2)
```

这比单纯惩罚动作绝对值更适合保留稳定的夹持动作。

### 3.6 成功终止

当指腹接触连续保持 50 个控制步时：

```text
success_reward = 40.0
terminated = True
```

如果 300 个控制步内没有完成，则 episode 截断，不能算成功。

## 4. 训练过程

每根手指使用 PPO 训练 100,000 步。由于 PPO 按 2,048 步 rollout 更新，实际保存步数为 100,352 步。

训练命令示例：

```powershell
.\.conda\python.exe rl\train_opposition.py --target 0 --timesteps 100000
.\.conda\python.exe rl\train_opposition.py --target 1 --timesteps 100000
.\.conda\python.exe rl\train_opposition.py --target 2 --timesteps 100000
.\.conda\python.exe rl\train_opposition.py --target 3 --timesteps 100000
```

模型保存位置：

```text
rl/checkpoints/aero_opposition_ppo_finger0.zip
rl/checkpoints/aero_opposition_ppo_finger1.zip
rl/checkpoints/aero_opposition_ppo_finger2.zip
rl/checkpoints/aero_opposition_ppo_finger3.zip
```

## 5. 最终评估结果

每个模型使用确定性策略进行 10 回合评估：

| 目标手指 | 成功率 | 典型 site 中心距离 | 指腹 geom 间隙 | 接触保持 |
| --- | ---: | ---: | ---: | ---: |
| 食指 | 10/10 | 约 12.8 mm | 约 1.3 mm | 50 步 |
| 中指 | 10/10 | 约 17.3 mm | 约 1.7 mm | 50 步 |
| 无名指 | 10/10 | 约 10.5 mm | 约 1.1 mm | 50 步 |
| 小指 | 10/10 | 约 29.0 mm | 约 2.0 mm | 50 步 |

小指的 site 中心距离相对较大，但两个指腹碰撞盒间隙约 2 mm，并且 MuJoCo 检测到真实接触，因此以 geom 间隙和接触保持作为最终判据。

## 6. 可视化回放

使用 `rl/replay_opposition.py` 可以查看单个回合的三视图变形过程：

```powershell
.\.conda\python.exe rl\replay_opposition.py `
  --model rl\checkpoints\aero_opposition_ppo_finger1 `
  --target 1 `
  --seed 0
```

窗口包含斜视、侧视和俯视三视图，并提供进度条、播放/暂停、上一帧、下一帧和回到开头。状态栏显示控制步、目标距离、geom 接触数、保持步数和成功标志。

## 7. 验证记录和结论

- `rl/opposition_env.py`、`rl/train_opposition.py`、`rl/evaluate_opposition.py` 和 `rl/replay_opposition.py` 已通过 `py_compile`。
- 环境已通过 Gymnasium `check_env` 检查。
- 四个固定目标均达到 10/10，证明对指预训练阶段已具备可用的基础策略。
- 当前检查点只用于对指动作，不代表球体夹持和抬升已经完成。
- 所有修改和模型均保存在本地，没有自动推送 GitHub。

