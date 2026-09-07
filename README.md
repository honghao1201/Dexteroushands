# Dexteroushands

基于 MuJoCo 的 Shadow Hand 灵巧手仿真与强化学习项目。当前主线使用 MuJoCo Menagerie 中的 Shadow Hand E3M5 右手模型，目标是先完成稳定的预抓取位姿，再训练手指闭合、真实接触、夹持和抬升。

TetherIA Aero Hand 文件仍保留在 `models/tetheria_aero_hand_open/`，用于历史实验和旧策略对照，不再作为当前 Shadow Hand 策略的运行模型。Aero Hand 和 Shadow Hand 的关节、执行器、坐标系不同，checkpoint 不能混用。

## 当前状态

当前可运行的 Shadow Hand 阶段如下：

| 阶段 | 场景 | 状态 | 最近验证 |
| --- | --- | --- | --- |
| PoseApproach | `models/shadow_hand/scene_pose.xml` | 已完成的水平姿态接近基线 | 5/5 成功，中心误差 2.4 mm，角度误差 0.61° |
| TopDownApproach | `models/shadow_hand/scene_topdown.xml` | 当前主线，从上方接近球体 | 5/5 成功，中心误差 6.4 mm，角度误差 1.41° |
| Finger Closure | 尚未建立 Shadow Hand 环境 | 待办 | 尚未验证 |
| Contact / Squeeze | 尚未建立 | 待办 | 尚未验证 |
| Lift / Hold | 尚未建立 | 待办 | 尚未验证 |

TopDownApproach 只负责把张开的手移动到球体上方的预抓取位姿，不闭合手指，也不抬升球体。

## 环境安装

在 VS Code 终端进入项目目录：

```powershell
cd D:\Projects\Dexteroushands
.\.conda\python.exe -m pip install -r requirements.txt
```

当前依赖包括 MuJoCo、Gymnasium、Stable-Baselines3 和 TensorBoard。项目使用 `.conda\python.exe`，不要把系统 Python 和项目环境混用。

## 查看 Shadow Hand 场景

`simulate.py` 保留了 Aero 兼容默认值。查看当前 Shadow Hand 顶部接近场景时显式指定 XML：

```powershell
.\.conda\python.exe simulate.py --scene models\shadow_hand\scene_topdown.xml
```

也可以直接启动 MuJoCo viewer：

```powershell
.\.conda\python.exe -m mujoco.viewer --mjcf models\shadow_hand\scene_topdown.xml
```

## Shadow Hand 模型

| 文件 | nq / nv | 执行器 | 用途 |
| --- | ---: | ---: | --- |
| `right_hand.xml` | 24 / 24 | 20 | 原始 Shadow Hand 模型，包含 24 个手部关节和 4 条手指腱绳 |
| `right_hand_pose.xml` | 28 / 28 | 24 | 增加整手平移和 yaw 的接近模型 |
| `scene_pose.xml` | 28 / 28 | 24 | PoseApproach 训练场景，包含球体、支撑面和光照 |
| `scene_topdown.xml` | 29 / 29 | 25 | 当前主线，额外增加 `hand_pitch`，使掌面旋转为朝下 |

`scene_topdown.xml` 的 5 个根部控制量为：

```text
hand_lift  世界 Z 方向平移
hand_x     世界 X 方向平移
hand_y     世界 Y 方向平移
hand_pitch 掌面俯仰，目标为 -pi/2
hand_yaw   绕世界竖直轴旋转
```

原生 Shadow Hand 手指保持张开，暂时不由 TopDownApproach 策略控制。

## 当前主线：从上方接近球体

训练环境为 `rl/shadow_topdown_approach_env.py`。策略输出 5 维根部残差动作，目标位姿由球心和 Shadow Hand 的 `approach_anchor` 解析计算：

- 掌面法向接近世界坐标 `-Z`；
- 锚点位于球心上方 100 mm；
- 手整体沿世界 Z 方向下降到预抓取位置；
- 保持手指张开，避免进入球体；
- 成功后连续保持 15 个控制步。

奖励包含锚点位置误差、掌面朝向、位置和姿态进展、目标间隙、穿透惩罚和动作幅度惩罚。成功条件为中心误差小于 8 mm、掌面角度误差小于 0.10 rad、掌面朝下程度大于 0.98，且手部与球体没有明显穿透。

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

回放窗口显示 `OBLIQUE`、`TOP`、`SIDE` 三个视角，支持拖动进度条、播放/暂停、上一帧、下一帧和键盘左右方向键。

## 已完成的水平接近基线

如果需要检查之前的水平姿态接近策略：

```powershell
.\.conda\python.exe rl\evaluate_shadow_pose_approach.py --model rl\checkpoints\shadow_pose_approach_ppo_v1 --episodes 10
.\.conda\python.exe rl\replay_shadow_pose_approach.py --model rl\checkpoints\shadow_pose_approach_ppo_v1
```

该策略使用 `scene_pose.xml`，掌面法向主要沿水平 X 方向。它用于验证根部平移和姿态对齐，不是当前从上方抓取任务的最终入口。

## 后续强化学习课程

后续任务按以下顺序进行：

1. 以 TopDownApproach 的终止状态作为初始状态，训练拇指和四指逐步闭合；
2. 只在真实手指-球体接触出现后增加夹紧奖励；
3. 检查接触法向力、切向力和摩擦裕量，建立稳定夹持；
4. 只有稳定夹持成立后才开放 `hand_lift`，沿世界 Z 轴抬升；
5. 约束球体滑移、旋转、支撑面接触和掉落；
6. 最后加入球体位置、摩擦系数、质量和视觉姿态随机化。

当前还没有完成 Shadow Hand 的闭合、夹持和抬升策略，因此不能把现有 TopDown checkpoint 当作完整抓取策略。

## TensorBoard

```powershell
.\.conda\Scripts\tensorboard.exe --logdir runs
```

打开 `http://localhost:6006` 查看训练曲线。

## 文件和版本管理

- 项目记录：[PROJECT.md](PROJECT.md)
- 任务清单：[TASKS.md](TASKS.md)
- Shadow Hand 顶部接近说明：[rl/SHADOW_TOPDOWN_APPROACH_CN.md](rl/SHADOW_TOPDOWN_APPROACH_CN.md)
- Shadow Hand 模型说明：[models/shadow_hand/README.md](models/shadow_hand/README.md)
- 训练 checkpoint 位于 `rl/checkpoints/`，默认被 `.gitignore` 排除；
- 训练日志和 TensorBoard 输出位于 `runs/`，默认被 `.gitignore` 排除；
- Git 默认只做本地提交，不会自动 push 到 GitHub。

手动同步时再执行：

```powershell
git status
git add README.md PROJECT.md TASKS.md rl models
git commit -m "docs: update Shadow Hand project workflow"
git push origin main
```

