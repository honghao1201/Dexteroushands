"""Shadow PoseApproach 阶段三视图进度条回放工具。

窗口默认显示斜视、掌面正视、侧视三个视角，可拖动进度条反复检查手腕接近过程。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tkinter as tk

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageTk
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from shadow_pose_approach_env import ShadowPoseApproachEnv
except ModuleNotFoundError:
    from rl.shadow_pose_approach_env import ShadowPoseApproachEnv


def resolve_model_path(model: pathlib.Path) -> pathlib.Path:
    if model.is_absolute():
        return model
    if (pathlib.Path.cwd() / model).exists() or (pathlib.Path.cwd() / f"{model}.zip").exists():
        return pathlib.Path.cwd() / model
    return pathlib.Path(__file__).resolve().parent / model


def collect_rollout(model_path: pathlib.Path, seed: int):
    """使用与评估相同的 VecNormalize 统计量，记录每一帧的物理状态。"""
    base = DummyVecEnv([lambda: ShadowPoseApproachEnv()])
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
    reset_info = {"success": False, "anchor_error": float("nan"), "angle_error": float("nan")}
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
        # 每个视窗使用更大的分辨率，便于看清 Shadow Hand 的掌面和指尖。
        self.renderer = mujoco.Renderer(sim.model, height=280, width=320)
        self.cameras = []
        # Shadow Hand 的掌面法向大致沿世界 +X：
        # 斜视用于观察整体构型，掌面正视用于检查五指相对位置，侧视用于检查接近深度。
        lookat = [0.20, -0.03, -0.045]
        for azimuth, elevation in ((225, -12), (180, 0), (270, 0)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = 0.55
            camera.lookat[:] = lookat
            self.cameras.append(camera)

        self.root = tk.Tk()
        self.root.title("Shadow PoseApproach 阶段三视图进度回放")
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
        for camera, name in zip(self.cameras, ("OBLIQUE", "PALM", "SIDE")):
            self.renderer.update_scene(self.sim.data, camera=camera)
            view = Image.fromarray(self.renderer.render())
            # 仅增强回放图像的亮度和对比度，不改变 MuJoCo 中的物理材质参数。
            view = ImageEnhance.Brightness(view).enhance(1.35)
            view = ImageEnhance.Contrast(view).enhance(1.12)
            draw = ImageDraw.Draw(view)
            draw.rectangle((0, 0, 84, 22), fill=(24, 30, 40))
            draw.text((5, 3), name, fill=(255, 255, 255))
            views.append(view)
        # 使用浅色拼接背景，避免暗色 MuJoCo 背景让整张回放显得过黑。
        image = Image.new("RGB", (960, 280), (224, 228, 235))
        for column, view in enumerate(views):
            image.paste(view, (column * 320, 0))
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)
        self.status.configure(text=(
            f"帧 {self.index}/{len(self.frames)-1}    "
            f"水平预抓取误差: {float(info.get('anchor_error', float('nan'))):.4f} m    "
            f"方向误差: {float(info.get('angle_error', float('nan'))) * 180.0 / 3.1415926:.2f} deg    "
            f"穿透深度: {float(info.get('penetration_depth', 0.0)):.4f} m    "
            f"Shadow接近成功: {info.get('success', False)}"
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
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/shadow_pose_approach_ppo_v1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    model_path = resolve_model_path(args.model)
    sim, frames, info = collect_rollout(model_path, args.seed)
    print(f"已记录 {len(frames)} 帧，最终中心误差 {info.get('anchor_error', float('nan')):.4f} m，方向误差 {float(info.get('angle_error', float('nan'))) * 180.0 / 3.1415926:.2f} deg，成功={info.get('success', False)}")
    ReplayWindow(sim, frames, fps=args.fps).run()


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()


