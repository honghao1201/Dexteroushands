# Dexteroushands

基于 MuJoCo 的 TetherIA Aero Hand 腱绳驱动灵巧手仿真与强化学习项目。

## 在 VS Code 中打开

打开 `D:\Projects\Dexteroushands`，在 Codex/终端中使用项目里的 Python 环境：

```powershell
.\.conda\python.exe -m pip install -r requirements.txt
```

## MuJoCo 手部仿真

```powershell
.\.conda\python.exe simulate.py
```

也可以直接查看原始手部场景：

```powershell
.\.conda\python.exe -m mujoco.viewer --mjcf models/tetheria_aero_hand_open/scene_right.xml
```

模型文件位于 `models/tetheria_aero_hand_open/`。当前训练场景固定掌面姿态为竖直方向，球体位于拇指一侧和其余四指一侧之间；`hand_lift` 滑动关节负责整只手沿世界坐标 Z 轴抬升。

## PPO 抓取任务

动作空间为 8 维：食指、中指、无名指、小指 4 个腱绳目标，拇指外展，拇指 2 个腱绳目标，以及整手抬升。环境文件为 `rl/aero_grasp_env.py`，场景文件为 `rl/grasp_scene.xml`。

训练 100,000 步：

```powershell
.\.conda\python.exe rl\train_ppo.py --timesteps 100000
```

模型输出：`rl/checkpoints/aero_grasp_ppo.zip`。

成功条件同时检查球体离开支撑面、整手已抬升、拇指与其余手指保持对向接触、法向夹紧力、低水平/垂直/角速度，并连续保持 20 个控制步。

## 回放和训练曲线

打开 MuJoCo 回放窗口：

```powershell
.\.conda\python.exe rl\evaluate_policy.py --model rl\checkpoints\aero_grasp_ppo --episodes 5
```

查看 TensorBoard：

```powershell
.\.conda\Scripts\tensorboard.exe --logdir runs
```

然后打开 `http://localhost:6006`。

## Git 版本管理

本地修改默认只保留在本机，不会自动上传 GitHub。检查、提交和手动同步：

```powershell
git status
git add .
git commit -m "describe the change"
git push origin main
```

远程仓库：[honghao1201/Dexteroushands](https://github.com/honghao1201/Dexteroushands)
