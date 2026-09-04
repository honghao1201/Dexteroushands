# Dexteroushands

项目目标和技术栈尚待确定。目前包含 Codex 协作与项目管理基础文件。

## 在 VS Code 中开始
1. 使用“文件 → 打开文件夹”打开 `D:\Projects\Dexteroushands`。
2. 打开 Codex 侧边栏并按提示登录；也可在命令面板中执行 `Codex: Open Codex Sidebar`。
3. 在 Codex 中发送下面的启动指令。

```text
请先阅读 AGENTS.md、PROJECT.md 和 TASKS.md，概述当前项目状态和适用规则。
检查这些文件是否存在；如果能正确读取，记录 T002 的验证结果。
接着帮助我明确项目目标、首版功能和验收标准，再更新项目说明和任务清单。
```

## 日常协作
- 查看进度：“请根据 TASKS.md 汇报已完成、进行中、阻塞及下一步。”
- 开始任务：“请执行 Txxx，完成后验证结果并更新任务清单。”
- 新增需求：“请把以下需求拆成有验收标准的任务，并更新里程碑：……”
- 结束工作：“请同步项目进展、验证结果和下一次继续的位置。”

## 文件导航
- [项目说明与里程碑](PROJECT.md)
- [任务清单](TASKS.md)
- [Codex 协作规则](AGENTS.md)

## 运行与测试
尚无业务代码。技术栈确定后补充实际环境、依赖安装、启动和测试命令。

## 版本管理
仓库已初始化并连接到 GitHub `origin/main`。在项目目录执行 `git status` 检查状态。
日常更新默认只保留在本地；需要手动上传时执行 `git push`。

## Aero Hand MuJoCo 仿真

模型位于 `models/tetheria_aero_hand_open/`，包含右手场景、MJCF 文件和网格资源。模型通过空间腱绳、弹簧和滑轮实现腱绳驱动。

首次运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python simulate.py
```

也可以直接使用 MuJoCo viewer：

```powershell
python -m mujoco.viewer --mjcf models/tetheria_aero_hand_open/scene_right.xml
```

仿真入口会打印关节、腱绳和执行器数量，并打开右手场景。抓取环境和 PPO 训练入口见下方。

## PPO 抓取控制

第一版抓取环境位于 `rl/`：动作是 7 个腱绳/拇指外展执行器的归一化目标，奖励鼓励接近方块、接触并抬升方块。

```powershell
.conda\python.exe -m pip install -r requirements.txt
cd rl
..\.conda\python.exe train_ppo.py
```

这是用于验证动作空间、接触和奖励设计的基线。训练成功后，再加入物体位置、尺寸、摩擦和初始姿态的随机化，以及分阶段课程学习。

## 参考
- [Codex IDE 官方说明](https://learn.chatgpt.com/docs/codex/ide)
- [AGENTS.md 官方说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
