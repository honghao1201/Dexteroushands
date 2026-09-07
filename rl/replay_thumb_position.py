"""拇指定位阶段三视图进度条回放工具。

窗口默认显示斜视、侧视、顶视三个视角，可拖动进度条反复检查手腕接近过程。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tkinter as tk

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from thumb_position_env import AeroThumbPositionEnv
except ModuleNotFoundError:
    from rl.thumb_position_env import AeroThumbPositionEnv


def resolve_model_path(model: pathlib.Path) -> pathlib.Path:
    if model.is_absolute():
        return model
    if (pathlib.Path.cwd() / model).exists() or (pathlib.Path.cwd() / f"{model}.zip").exists():
        return pathlib.Path.cwd() / model
    return pathlib.Path(__file__).resolve().parent / model


def collect_rollout(model_path: pathlib.Path, seed: int):
    """使用与评估相同的 VecNormalize 统计量，记录每一帧的物理状态。"""
    base = DummyVecEnv([lambda: AeroThumbPositionEnv()])
    stats = pathlib.Path(str(model_path) + "_vecnormalize.pkl")
    if stats.exists():
        env = VecNormalize.load(str(stats), base)
        env.training = False
        env.norm_reward = False
    else:
        env = base
    policy = PPO.load(str(model_path), env=env, device="cpu")
    obs = env.reset()
    sim = env.envs[0].unwrapped
    reset_info = {"success": False, "anchor_error": float("nan")}
    frames = [(sim.data.qpos.copy(), sim.data.qvel.copy(), reset_info.copy())]
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        info = dict(infos[0])
        frames.append((sim.data.qpos.copy(), sim.data.qvel.copy(), info))
        if bool(done[0]):
            break
    env.close()
    return sim, frames, frames[-1][2]


class ReplayWindow:
    """三视图回放窗口：进度条可拖动，按钮和键盘均可控制。"""

    def __init__(self, sim, frames, fps: int = 20):
        self.sim = sim
        self.frames = frames
        self.fps = max(1, fps)
        self.index = 0
        self.playing = False
        self.renderer = mujoco.Renderer(sim.model, height=240, width=300)
        self.cameras = []
        lookat = [0.10, -0.03, -0.03]
        for azimuth, elevation in ((235, -18), (90, 0), (0, 90)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = 0.28
            camera.lookat[:] = lookat
            self.cameras.append(camera)

        self.root = tk.Tk()
        self.root.title("拇指定位阶段三视图进度回放")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=8, pady=8)
        self.slider = tk.Scale(self.root, from_=0, to=max(0, len(frames) - 1), orient=tk.HORIZONTAL,
                               length=900, showvalue=False, command=self.on_slider)
        self.slider.pack(fill=tk.X, padx=12)
        controls = tk.Frame(self.root)
        controls.pack(pady=6)
        tk.Button(controls, text="|< 开始", command=lambda: self.show_frame(0)).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="上一帧", command=lambda: self.show_frame(self.index - 1)).pack(side=tk.LEFT, padx=3)
        self.play_button = tk.Button(controls, text="播放", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="下一帧", command=lambda: self.show_frame(self.index + 1)).pack(side=tk.LEFT, padx=3)
        self.status = tk.Label(self.root, anchor="w", justify=tk.LEFT)
        self.status.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.show_frame(self.index - 1))
        self.root.bind("<Right>", lambda _event: self.show_frame(self.index + 1))
        self.show_frame(0)

    def render_current(self):
        qpos, qvel, info = self.frames[self.index]
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        mujoco.mj_forward(self.sim.model, self.sim.data)
        views = []
        for camera, name in zip(self.cameras, ("斜视", "侧视", "顶视")):
            self.renderer.update_scene(self.sim.data, camera=camera)
            view = Image.fromarray(self.renderer.render())
            draw = ImageDraw.Draw(view)
            draw.rectangle((0, 0, 60, 21), fill=(0, 0, 0))
            draw.text((5, 3), name, fill=(255, 255, 255))
            views.append(view)
        image = Image.new("RGB", (900, 240), (30, 30, 30))
        for column, view in enumerate(views):
            image.paste(view, (column * 300, 0))
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)
        self.status.configure(text=(
            f"帧 {self.index}/{len(self.frames)-1}    "
            f"拇指位置误差: {float(info.get('position_error', float('nan'))):.4f} m    "
            f"两侧关系: {info.get('aligned', False)}    "
            f"穿透深度: {float(info.get('penetration_depth', 0.0)):.4f} m    "
            f"Pregrasp成功: {info.get('success', False)}"
        ))

    def show_frame(self, index: int):
        self.index = max(0, min(int(index), len(self.frames) - 1))
        self.slider.set(self.index)
        self.render_current()

    def on_slider(self, value: str):
        index = int(float(value))
        if index != self.index:
            self.index = index
            self.render_current()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text="暂停" if self.playing else "播放")
        if self.playing:
            self.play_step()

    def play_step(self):
        if not self.playing:
            return
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self.play_button.configure(text="播放")
            return
        self.show_frame(self.index + 1)
        self.root.after(max(1, int(1000 / self.fps)), self.play_step)

    def close(self):
        self.playing = False
        self.renderer.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/aero_thumb_position_calibrated_v2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    model_path = resolve_model_path(args.model)
    sim, frames, info = collect_rollout(model_path, args.seed)
    print(f"已记录 {len(frames)} 帧，最终拇指位置误差 {info.get('position_error', float('nan')):.4f} m，成功={info.get('success', False)}")
    ReplayWindow(sim, frames, fps=args.fps).run()


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()



