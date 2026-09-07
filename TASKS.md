# 任务清单

状态只使用：待办、进行中、阻塞、已完成。每个已完成任务都必须有可运行命令或评估结果作为验收证据。

| 编号 | 任务 | 优先级 | 状态 | 验收标准 |
| --- | --- | --- | --- | --- |
| T001 | 建立本地项目管理文件 | P0 | 已完成 | `AGENTS.md`、`PROJECT.md`、`TASKS.md` 和 `README.md` 已建立并与当前 Shadow Hand 主线一致 |
| T002 | 在 VS Code 中验证 Codex 工作流 | P0 | 待用户验证 | 用户在 VS Code 中打开项目并确认 Codex、Python 环境和终端可用 |
| T003 | 保留 Aero Hand 历史基线 | P2 | 已完成 | Aero 模型、旧环境和旧 checkpoint 保留，明确与 Shadow Hand 隔离 |
| T004 | 导入 Shadow Hand E3M5 模型 | P0 | 已完成 | `right_hand.xml` 可加载，原生模型为 `nq=24、nu=20` |
| T005 | 建立 Shadow Hand Pose 场景 | P0 | 已完成 | `scene_pose.xml` 可加载，`nq=28、nu=24`，包含球体、支撑面和根部接近自由度 |
| T006 | 完成水平姿态接近基线 | P1 | 已完成 | `shadow_pose_approach_ppo_v1` 最近 5/5 成功，平均中心误差 2.4 mm |
| T007 | 建立 Shadow Hand 顶部接近模型 | P0 | 已完成 | `scene_topdown.xml` 增加 `hand_pitch`，可将掌面旋至朝下，`nq=29、nu=25` |
| T008 | 训练顶部预抓取策略 | P0 | 已完成 | `shadow_topdown_approach_ppo_v1` 最近 5/5 成功，平均中心误差 6.4 mm、角度误差 1.41° |
| T009 | 完成顶部接近三视图回放 | P1 | 已完成 | `replay_shadow_topdown_approach.py` 支持 OBLIQUE、TOP、SIDE、进度条、播放和逐帧查看 |
| T010 | 训练 Shadow Hand 手指闭合 | P0 | 待办 | 从 TopDownApproach 终止状态开始，拇指和四指逐步闭合，保持非穿透并通过评估 |
| T011 | 建立球体真实接触和夹紧奖励 | P0 | 待办 | 至少拇指和三根手指形成真实接触，记录法向力、切向力和摩擦裕量 |
| T012 | 训练稳定夹持 | P0 | 待办 | 球体相对手位移和角速度受限，连续保持指定步数，不依赖掌部托举 |
| T013 | 训练整手竖直抬升 | P0 | 待办 | 只有稳定夹持成立后开放 `hand_lift`，球体离开支撑面且不掉落 |
| T014 | 增加随机化和泛化评估 | P1 | 待办 | 随机化球体位置、质量、摩擦和初始姿态，报告多回合成功率 |
| T015 | 统一 Shadow Hand 训练日志 | P1 | 待办 | 每个阶段记录 reward 分项、成功判据、checkpoint、评估命令和回放文件 |
| T016 | 设计真实执行器接口 | P2 | 待办 | 完成 Shadow Hand 关节、腱绳、力传感器和安全限幅映射 |
| T017 | 本地 Git 提交和手动同步 | P1 | 待用户操作 | 用户确认后再执行 `git add`、`commit` 和 `push`，默认不自动上传 |

## 当前执行顺序

```text
T007/T008 顶部预抓取
        ↓
T010 手指闭合
        ↓
T011 真实接触与夹紧
        ↓
T012 稳定保持
        ↓
T013 整手抬升
        ↓
T014 随机化泛化
```

## 常用验收命令

```powershell
# 检查 Shadow Hand 顶部接近策略
.\.conda\python.exe rl\evaluate_shadow_topdown_approach.py --model rl\checkpoints\shadow_topdown_approach_ppo_v1 --episodes 10

# 查看三视图和进度条
.\.conda\python.exe rl\replay_shadow_topdown_approach.py --model rl\checkpoints\shadow_topdown_approach_ppo_v1

# 检查 MuJoCo 场景规模
.\.conda\python.exe -c "import mujoco; m=mujoco.MjModel.from_xml_path('models/shadow_hand/scene_topdown.xml'); print(m.nq,m.nv,m.njnt,m.nu,m.ntendon)"
```

更新本文件时同步更新 `PROJECT.md` 的当前状态和 `README.md` 的运行命令。Git 提交默认保留在本地，只有用户明确要求时才推送到 GitHub。

