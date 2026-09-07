"""带进度条的拇指对指轨迹回放工具。"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tkinter as tk

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from stable_baselines3 import PPO

try:
    from opposition_env import AeroOppositionEnv
except ModuleNotFoundError:  # 支持从项目根目录以 rl.replay_opposition 方式导入。
    from rl.opposition_env import AeroOppositionEnv


def resolve_model_path(model: pathlib.Path) -> pathlib.Path:
    """兼容从项目根目录或 rl 目录传入的模型路径。"""
    if model.is_absolute():
        return model
    if (pathlib.Path.cwd() / model).exists() or (pathlib.Path.cwd() / f"{model}.zip").exists():
        return pathlib.Path.cwd() / model
    return pathlib.Path(__file__).resolve().parent / model


def collect_rollout(env: AeroOppositionEnv, policy: PPO, seed: int):
    """先运行一回合并保存每个控制步的 qpos、qvel 和评估信息。"""
    observation, reset_info = env.reset(seed=seed)
    frames = [(env.data.qpos.copy(), env.data.qvel.copy(), reset_info)]
    info = reset_info
    while True:
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        frames.append((env.data.qpos.copy(), env.data.qvel.copy(), dict(info)))
        if terminated or truncated:
            break
    return frames, info


class ReplayWindow:
    """使用 Tkinter 滑块查看 MuJoCo 轨迹的每一个控制步。"""

    def __init__(self, env: AeroOppositionEnv, frames, fps: int = 20):
        self.env = env
        self.frames = frames
        self.fps = max(1, fps)
        self.index = 0
        self.playing = False
        # 模型默认 offwidth 为 640，每个视图使用 300x240，三视图横向拼接。
        self.renderer = mujoco.Renderer(env.model, height=240, width=300)
        self.cameras = []
        for azimuth, elevation in ((270, 0), (0, 0), (0, 90)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = 0.35
            camera.lookat[:] = [0.17, -0.02, -0.01]
            self.cameras.append(camera)

        self.root = tk.Tk()
        self.root.title("拇指-食指对指轨迹回放")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=8, pady=8)

        # 进度条可以拖动到任意控制步，反复查看局部变形过程。
        self.slider = tk.Scale(
            self.root,
            from_=0,
            to=len(frames) - 1,
            orient=tk.HORIZONTAL,
            length=900,
            showvalue=False,
            command=self.on_slider,
        )
        self.slider.pack(fill=tk.X, padx=12)

        controls = tk.Frame(self.root)
        controls.pack(pady=6)
        tk.Button(controls, text="|< 回到开头", command=lambda: self.show_frame(0)).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="上一帧", command=self.previous_frame).pack(side=tk.LEFT, padx=3)
        self.play_button = tk.Button(controls, text="播放", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="下一帧", command=self.next_frame).pack(side=tk.LEFT, padx=3)

        self.status = tk.Label(self.root, anchor="w", justify=tk.LEFT)
        self.status.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.previous_frame())
        self.root.bind("<Right>", lambda _event: self.next_frame())
        self.show_frame(0)

    def render_current(self) -> None:
        """把选中的 qpos 写回 MuJoCo，并渲染当前帧。"""
        qpos, qvel, info = self.frames[self.index]
        self.env.data.qpos[:] = qpos
        self.env.data.qvel[:] = qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        # 分别渲染斜视、侧视和俯视，再拼成一张同步画面。
        rendered_views = []
        view_names = ("斜视", "侧视", "俯视")
        for camera, name in zip(self.cameras, view_names):
            self.renderer.update_scene(self.env.data, camera=camera)
            view = Image.fromarray(self.renderer.render())
            ImageDraw.Draw(view).rectangle((0, 0, 58, 20), fill=(0, 0, 0))
            ImageDraw.Draw(view).text((5, 3), name, fill=(255, 255, 255))
            rendered_views.append(view)
        image = Image.new("RGB", (900, 240), (30, 30, 30))
        for column, view in enumerate(rendered_views):
            image.paste(view, (column * 300, 0))
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)
        self.status.configure(
            text=(
                f"控制步: {self.index}/{len(self.frames) - 1}    "
                f"目标距离: {info.get('target_distance', 0.0):.4f} m    "
                f"尖端接触数: {info.get('contacts', 0)}    "
                f"保持步数: {info.get('hold_steps', 0)}    "
                f"成功: {info.get('success', False)}"
            )
        )

    def show_frame(self, index: int) -> None:
        """显示指定帧，并同步进度条位置。"""
        self.index = max(0, min(int(index), len(self.frames) - 1))
        self.slider.set(self.index)
        self.render_current()

    def on_slider(self, value: str) -> None:
        """响应用户拖动进度条。"""
        if int(float(value)) != self.index:
            self.index = int(float(value))
            self.render_current()

    def previous_frame(self) -> None:
        self.show_frame(self.index - 1)

    def next_frame(self) -> None:
        self.show_frame(self.index + 1)

    def toggle_play(self) -> None:
        """开始或暂停自动播放。"""
        self.playing = not self.playing
        self.play_button.configure(text="暂停" if self.playing else "播放")
        if self.playing:
            self.play_step()

    def play_step(self) -> None:
        """按设定帧率推进回放，播放到末尾后自动暂停。"""
        if not self.playing:
            return
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self.play_button.configure(text="播放")
            return
        self.show_frame(self.index + 1)
        self.root.after(max(1, int(1000 / self.fps)), self.play_step)

    def close(self) -> None:
        """释放渲染器和 MuJoCo 环境。"""
        self.playing = False
        self.renderer.close()
        self.env.close()
        self.root.destroy()

    def run(self) -> None:
        """进入可视化窗口事件循环。"""
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_opposition_ppo_finger0"))
    parser.add_argument("--target", type=int, choices=range(4), default=0, help="目标手指，0 表示食指")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20, help="自动播放帧率")
    args = parser.parse_args()

    env = AeroOppositionEnv(target_index=args.target)
    policy = PPO.load(resolve_model_path(args.model), env=env)
    frames, final_info = collect_rollout(env, policy, args.seed)
    print(
        f"已记录 {len(frames)} 帧，目标={final_info['target_name']}，"
        f"success={final_info['success']}，contacts={final_info['contacts']}"
    )
    ReplayWindow(env, frames, fps=args.fps).run()


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
