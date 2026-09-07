# Dexteroushands 项目记录

## 项目定位

本项目使用 MuJoCo 和 Stable-Baselines3 训练 Shadow Hand 灵巧手的分阶段抓取策略。当前主线不是直接从张开的手训练完整抓球，而是先把根部位姿、预抓取位置、手指闭合、真实接触、夹持和抬升拆成独立阶段。

TetherIA Aero Hand 仍保留在项目中作为历史基线。它的模型和 checkpoint 与 Shadow Hand 不兼容，后续文档和新策略以 Shadow Hand 为准。

## 当前目标

让 Shadow Hand 以掌面朝下的姿态，从球体上方接近到安全预抓取位置，然后逐步完成：

1. 手指保持张开，整手从上方下降；
2. 拇指与四指围绕球体形成预抓取构型；
3. 手指闭合并建立真实接触；
4. 通过法向夹紧力产生摩擦，稳定保持球体；
5. 整只手沿世界 Z 轴向上抬升球体；
6. 抬升过程中避免滑移、明显旋转、支撑面接触和掉落。

## 当前主模型

模型来自 Shadow Hand E3M5，位于 `models/shadow_hand/`。当前实际使用的模型规模如下：

| 场景 | nq / nv | njnt | nu | 说明 |
| --- | ---: | ---: | ---: | --- |
| `right_hand.xml` | 24 / 24 | 24 | 20 | 原生 Shadow Hand |
| `scene_pose.xml` | 28 / 28 | 28 | 24 | 4 个根部接近自由度 |
| `scene_topdown.xml` | 29 / 29 | 29 | 25 | 在 Pose 模型上增加 `hand_pitch` |

`scene_topdown.xml` 是当前策略主线。它增加了 `hand_pitch`，目标值为 `-pi/2`，使掌面法向朝向世界 `-Z`；`hand_lift` 仍表示世界 Z 方向平移，后续 Lift 阶段将复用这个自由度。

## 策略阶段

### PoseApproach：水平姿态接近基线

文件：

- 环境：`rl/shadow_pose_approach_env.py`
- 场景：`models/shadow_hand/scene_pose.xml`
- checkpoint：`rl/checkpoints/shadow_pose_approach_ppo_v1.zip`
- 回放：`rl/replay_shadow_pose_approach.py`

该阶段控制整手的平移和 yaw，手指保持张开，用于验证锚点定位、水平姿态约束和穿透惩罚。它是已完成的对照基线，不是当前顶部接近任务的最终入口。

### TopDownApproach：当前主线

文件：

- 环境：`rl/shadow_topdown_approach_env.py`
- 场景：`models/shadow_hand/scene_topdown.xml`
- 训练：`rl/train_shadow_topdown_approach.py`
- 评估：`rl/evaluate_shadow_topdown_approach.py`
- 回放：`rl/replay_shadow_topdown_approach.py`
- checkpoint：`rl/checkpoints/shadow_topdown_approach_ppo_v1.zip`

动作空间是 5 维根部残差：`hand_lift`、`hand_x`、`hand_y`、`hand_pitch` 和 `hand_yaw`。目标位姿由球心和 `approach_anchor` 计算，不让 PPO 同时猜测坐标系和目标位置。

主要奖励项：

- 锚点到球体上方目标的指数位置奖励；
- 掌面法向朝向 `-Z` 的姿态奖励；
- 位置误差和姿态误差的逐步进展奖励；
- 预抓取安全间隙奖励；
- 穿透、过大间隙和动作幅度惩罚。

成功判据：

- 锚点中心误差小于 8 mm；
- 掌面角度误差小于 0.10 rad；
- 掌面朝下程度大于 0.98；
- 手部与球体没有明显穿透；
- 连续保持 15 个控制步。

## 最近验证

2026-09-07 使用确定性策略重新验证：

- `shadow_pose_approach_ppo_v1`：5/5 成功，平均中心误差 2.4 mm，平均角度误差 0.61°；
- `shadow_topdown_approach_ppo_v1`：5/5 成功，平均中心误差 6.4 mm，平均角度误差 1.41°，最小间隙保持为非穿透。

这些结果只证明两个接近阶段有效，不代表已经完成手指闭合、摩擦夹持或抬升。

## 物理和任务约束

1. TopDownApproach 阶段手掌法向必须朝向世界 `-Z`，不通过手腕横向旋转来替代下降动作。
2. 手整体通过根部平移接近和移动；不能只移动手指而把根部留在原地。
3. 预抓取阶段禁止手部穿入球体；接触阶段才允许使用真实接触建立夹紧力。
4. Lift 阶段必须由整只手和手腕沿世界 Z 轴上移，不能用掌部或手指从下方托球。
5. 稳定保持需要同时检查球体相对手的位移、线速度、角速度、支撑面接触和掉落。

## 未完成部分

- Shadow Hand 手指闭合和拇指对向构型；
- 球体真实接触、法向力、切向力和摩擦裕量；
- 稳定夹持后的整手抬升；
- 物体质量、摩擦和位置随机化；
- 统一 Shadow Hand 的长回合评估和训练日志；
- 真实执行器映射和 sim-to-real 接口。

## 版本管理约定

本地修改和验证优先，默认不执行 `git push`。训练 checkpoint、TensorBoard 日志和临时回放图片不纳入默认提交；代码、模型 XML、README、PROJECT 和 TASKS 在确认后由用户手动提交和同步。

