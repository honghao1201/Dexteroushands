"""多指抓取三视图进度条回放工具。"""

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
    from multifinger_grasp_env import AeroMultiFingerGraspEnv
except ModuleNotFoundError:
    from rl.multifinger_grasp_env import AeroMultiFingerGraspEnv


def resolve_model_path(model: pathlib.Path) -> pathlib.Path:
    """兼容从项目根目录或 rl 目录传入的模型路径。"""
    if model.is_absolute():
        return model
    if (pathlib.Path.cwd() / model).exists() or (pathlib.Path.cwd() / f"{model}.zip").exists():
        return pathlib.Path.cwd() / model
    return pathlib.Path(__file__).resolve().parent / model


def collect_rollout(env: AeroMultiFingerGraspEnv, policy: PPO, seed: int):
    """记录一回合的 qpos、qvel 和诊断信息，供滑块逐帧回放。"""
    observation, reset_info = env.reset(seed=seed)
    frames = [(env.data.qpos.copy(), env.data.qvel.copy(), dict(reset_info))]
    info = reset_info
    while True:
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        frames.append((env.data.qpos.copy(), env.data.qvel.copy(), dict(info)))
        if terminated or truncated:
            break
    return frames, info


class ReplayWindow:
    """用滑块、按钮和键盘控制多指抓取三视图回放。"""

    def __init__(self, env: AeroMultiFingerGraspEnv, frames, fps: int = 20):
        self.env = env
        self.frames = frames
        self.fps = max(1, fps)
        self.index = 0
        self.playing = False
        # 每个视图 300x240，合成宽度 900，避免超过 MuJoCo 默认 framebuffer。
        self.renderer = mujoco.Renderer(env.model, height=240, width=300)
        lookat = [0.10, -0.03, -0.03] if env.freeze_object else [0.127, -0.015, -0.02]
        self.cameras = []
        for azimuth, elevation in ((270, 0), (0, 0), (0, 90)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = 0.28
            camera.lookat[:] = lookat
            self.cameras.append(camera)

        self.root = tk.Tk()
        self.root.title("多指球体夹紧三视图回放")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=8, pady=8)
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
        tk.Button(controls, text="|< 回到开始", command=lambda: self.show_frame(0)).pack(side=tk.LEFT, padx=3)
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
        """渲染当前 qpos，并合成斜视、侧视和顶视画面。"""
        qpos, qvel, info = self.frames[self.index]
        self.env.data.qpos[:] = qpos
        self.env.data.qvel[:] = qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        rendered = []
        for camera, name in zip(self.cameras, ("斜视", "侧视", "顶视")):
            self.renderer.update_scene(self.env.data, camera=camera)
            view = Image.fromarray(self.renderer.render())
            draw = ImageDraw.Draw(view)
            draw.rectangle((0, 0, 58, 20), fill=(0, 0, 0))
            draw.text((5, 3), name, fill=(255, 255, 255))
            rendered.append(view)
        image = Image.new("RGB", (900, 240), (30, 30, 30))
        for column, view in enumerate(rendered):
            image.paste(view, (column * 300, 0))
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)
        gaps = info.get("gaps", [])
        min_gap = min(gaps) if len(gaps) else 0.0
        self.status.configure(
            text=(
                f"控制步 {self.index}/{len(self.frames) - 1}    "
                f"接触手指数: {info.get('finger_touch_count', 0)} + 拇指 {int(info.get('thumb_touch', 0))}    "
                f"最小指尖间隙: {min_gap:.4f} m    "
                f"相对两侧: {info.get('opposite_side', False)}    "
                f"保持: {info.get('hold_steps', 0)}    "
                f"成功: {info.get('success', False)}"
            )
        )

    def show_frame(self, index: int) -> None:
        self.index = max(0, min(int(index), len(self.frames) - 1))
        self.slider.set(self.index)
        self.render_current()

    def on_slider(self, value: str) -> None:
        index = int(float(value))
        if index != self.index:
            self.index = index
            self.render_current()

    def previous_frame(self) -> None:
        self.show_frame(self.index - 1)

    def next_frame(self) -> None:
        self.show_frame(self.index + 1)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="暂停" if self.playing else "播放")
        if self.playing:
            self.play_step()

    def play_step(self) -> None:
        if not self.playing:
            return
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self.play_button.configure(text="播放")
            return
        self.show_frame(self.index + 1)
        self.root.after(max(1, int(1000 / self.fps)), self.play_step)

    def close(self) -> None:
        self.playing = False
        self.renderer.close()
        self.env.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_multifinger_static_v2_smoketest"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--hard-scene", action="store_true")
    parser.add_argument("--use-prior", action="store_true")
    args = parser.parse_args()
    position = (0.127, -0.015, -0.064) if args.hard_scene or args.dynamic else (0.10, -0.03, -0.03)
    env = AeroMultiFingerGraspEnv(
        freeze_object=not args.dynamic,
        use_prior=args.use_prior,
        object_position=position,
    )
    policy = PPO.load(resolve_model_path(args.model), env=env, device="cpu")
    frames, final_info = collect_rollout(env, policy, args.seed)
    print(
        f"已记录 {len(frames)} 帧，接触手指数={final_info.get('finger_touch_count')}，"
        f"拇指={int(final_info.get('thumb_touch', 0))}，成功={final_info.get('success')}"
    )
    ReplayWindow(env, frames, fps=args.fps).run()


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
