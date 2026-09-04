# Dexteroushands

## 项目目标
待用户说明。当前仅完成项目管理基础文件的准备，不代表业务项目已实现。

## 当前状态
- 项目目录：`D:\Projects\Dexteroushands`
- 初始检查：空目录，无业务代码。
- 编辑器：VS Code；官方 Codex 扩展已安装，编辑器内登录与实际对话待验证。
- 技术栈：Python + MuJoCo；模型为 TetherIA Aero Hand Open MJCF。
- 当前运行环境：需要 Python 3 和 `mujoco` Python 包；本机尚未安装 Python。
- 版本管理：已初始化本地 Git 仓库，当前分支为 `main`，首个基线提交为 `83c9d25`。

## 范围与验收
- 目标用户：待确定。
- 核心功能：待确定。
- 首个可运行版本的验收标准：能加载右手场景并打开 MuJoCo viewer，打印模型的关节、腱绳和执行器数量。
- 暂不纳入的功能：待确定。

## 里程碑
| 编号 | 里程碑 | 完成条件 | 状态 |
| --- | --- | --- | --- |
| M0 | 协作环境就绪 | 管理文件到位，VS Code 中 Codex 能读取项目规则并回答 | 进行中 |
| M1 | 明确项目范围 | 明确目标、首版功能及验收标准 | 待办 |
| M2 | 首个可运行版本 | `simulate.py` 加载右手场景并启动 viewer | 已完成 |
| M3 | 对向夹持抬升基线 | PPO 在腱绳空间训练，模型可回放并报告抓取指标 | 已完成（100,352 步基线；需继续评估成功率） |

## 已有决策
- 使用项目内 Markdown 文件维护任务、里程碑和协作规则。
- Git 提交身份：Hao `<honghao19941201@gmail.com>`。
- 先确认实际需求，再选择技术栈和搭建代码结构。
- 尚未设置交付日期。
- 采用 TetherIA Aero Hand Open 的 MuJoCo Menagerie 模型，模型文件保存在 `models/tetheria_aero_hand_open/`。
- 仿真启动入口为 `simulate.py`，依赖记录在 `requirements.txt`。
- 抓取环境为 `rl/aero_grasp_env.py`，训练入口为 `rl/train_ppo.py`，最近模型为 `rl/checkpoints/aero_grasp_ppo.zip`。
- 抓取奖励包含掌心固定、逐步闭合、拇指与四指对向接触、抬升和低水平滑移速度约束。

## 下一步
安装 Python 并运行 `python simulate.py`；随后添加腱绳控制器、状态观测和第一个操作任务。
