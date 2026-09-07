# Pregrasp 奖励逐项审查

## 当前实现（`rl/pregrasp_env.py`）

| 项目 | 当前公式/权重 | 物理含义 |
|---|---|---|
| 四指侧向距离 | `3.0 * finger_side_score` | 四指位于球体 +Y 侧的连续距离分数 |
| 拇指侧向距离 | `3.0 * thumb_side_score` | 拇指位于球体 -Y 侧的连续距离分数 |
| 四指球面间隙 | `2.0 * finger_gap_score` | 四指接近各自目标间隙 |
| 拇指球面间隙 | `3.0 * thumb_gap_score` | 拇指接近目标间隙，单独提高权重 |
| 掌面姿态 | `1.0 * palm_horizontal` | 保持掌面竖直 |
| 间隙进步 | `0.5 * (上一时刻误差-当前误差)` | 奖励本步真实改善 |
| 动作平滑 | `-0.02 * mean((action-last_action)^2)` | 抑制抖动和突然闭合 |
| 穿透惩罚 | `-100.0 * sum(max(0,-gap))` | 惩罚指尖进入球体 |
| 穿透终止 | `max(penetration)>0.003` | 穿透超过 3 mm 立即结束 |
| 成功保持 | 至少三指和拇指进入间隙窗口，并保持 25 步 | 防止瞬间接近被误判为成功 |

## 当前奖励的主要问题

1. `tip_error` 仍然是三维指尖点误差，尤其拇指的目标高度和路径可能超出当前机构可达范围。Pregrasp 更适合奖励“球面间隙 + 侧向符号”，而不是要求每个指尖到达一个精确 3D 点。
2. `opposite_side` 是二值奖励。手指从正确一侧移动到错误一侧时，奖励变化不连续，PPO 很难知道应该往哪个方向修正。
3. `gap_progress` 的正奖励必须与穿透惩罚配套，否则快速穿过球面可能获得暂时的进步分。
4. 目前没有单独的“拇指接近奖励”。四指和拇指的可达性差异很大，建议分开记录和加权。
5. 所有执行器共享相近的奖励尺度，但拇指动作对任务更关键，建议增加拇指侧向距离和拇指间隙的独立项。

## 可参考的开源实现

- [robosuite Stack](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/stack.py)：把 reaching、grasping、lifting、aligning、stacking 分成阶段，并使用阶段门控；它的 reaching 使用 `1-tanh(k*distance)`，grasping 使用严格接触检查。
- [robosuite Lift](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/lift.py)：把 dense shaping 与 sparse success 分开，成功条件采用明确的几何阈值。
- [drl_grasping](https://github.com/AndrejOrsula/drl_grasping)：将 Reach-v0 和 Grasp-v0 分成独立任务，并通过 `GraspCurriculum` 逐步增加难度。
- [Bi-DexHands](https://github.com/PKU-MARL/DexterousHands/blob/main/docs/environments.md)：常用指数型目标距离奖励，目标误差越小奖励越快上升，同时把手部基座平移作为单独动作通道。
- [DexGraspBench](https://github.com/JYChen18/DexGraspBench/)：把仿真成功率、力闭合、穿透深度和接触质量作为独立评估指标，而不是全部混成一个奖励。

## 建议讨论的三套方案

### 方案 A：几何接近型（推荐先试）

```text
r = 3.0 * side_distance_score
  + 3.0 * thumb_gap_score
  + 2.0 * finger_gap_score
  + 1.0 * palm_orientation
  + 0.5 * progress
  - 100.0 * penetration
  - 0.02 * action_change
```

成功条件：拇指间隙和至少三根手指间隙进入安全窗口，两侧关系正确，保持 25 步。完全删除精确 3D 指尖目标。

### 方案 B：接触准备型

在方案 A 基础上增加：

```text
+ 2.0 * contact_count / 5
+ 1.0 * normal_force_in_target_range
```

适合下一阶段 Contact/Squeeze，但 Pregrasp 阶段可能过早鼓励碰球。

### 方案 C：分阶段门控型

```text
未满足两侧关系：只奖励侧向定位，不奖励闭合
满足两侧关系后：开放间隙接近奖励
出现真实接触后：开放法向力和摩擦奖励
稳定夹持后：才开放抬升奖励
```

这是最接近 robosuite 和 DexGraspBench 的做法，奖励更容易解释，但需要额外的阶段状态机。
