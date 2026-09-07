"""Shadow Hand 顶部接近阶段的三视图进度条回放。"""
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
    from shadow_topdown_approach_env import ShadowTopDownApproachEnv
except ModuleNotFoundError:
    from rl.shadow_topdown_approach_env import ShadowTopDownApproachEnv


def resolve_model_path(model: pathlib.Path) -> pathlib.Path:
    if model.is_absolute():
        return model
    if (pathlib.Path.cwd() / model).exists() or (pathlib.Path.cwd() / f"{model}.zip").exists():
        return pathlib.Path.cwd() / model
    return pathlib.Path(__file__).resolve().parent / model


def collect_rollout(model_path: pathlib.Path, seed: int):
    """按评估配置记录每一帧 qpos/qvel，供进度条拖动查看。"""
    base = DummyVecEnv([lambda: ShadowTopDownApproachEnv()])
    stats = pathlib.Path(str(model_path) + "_vecnormalize.pkl")
    env = VecNormalize.load(str(stats), base) if stats.exists() else base
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False
    policy = PPO.load(str(model_path), env=env, device="cpu")
    obs = env.reset()
    sim = env.envs[0].unwrapped
    frames = [(sim.data.qpos.copy(), sim.data.qvel.copy(), {"success": False})]
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        info = dict(infos[0])
        if bool(done[0]) and "terminal_qpos" in info:
            # VecEnv 已经把环境 reset 了，这里改用环境在 done 前保存的真实状态。
            qpos = np.asarray(info["terminal_qpos"], dtype=np.float64).copy()
            qvel = np.asarray(info["terminal_qvel"], dtype=np.float64).copy()
        else:
            qpos = sim.data.qpos.copy()
            qvel = sim.data.qvel.copy()
        frames.append((qpos, qvel, info))
        if bool(done[0]):
            break
    info = frames[-1][2]
    # 不在这里调用 env.close()：DummyVecEnv 关闭时可能重置 qpos，回放需要保留最后一帧状态。
    return sim, frames, info


class ReplayWindow:
    """三视图回放窗口，包含可拖动进度条、播放和逐帧按钮。"""

    def __init__(self, sim, frames, fps: int = 20):
        self.sim = sim
        self.frames = frames
        self.fps = max(1, fps)
        self.index = 0
        self.playing = False
        self.renderer = mujoco.Renderer(sim.model, height=280, width=320)
        self.cameras = []
        lookat = [0.10, -0.03, 0.02]
        # 顶部接近阶段重点观察：整体斜视、俯视手指与球的相对位置、侧面下降高度。
        for azimuth, elevation in ((225, -15), (0, -70), (270, 0)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = 0.60 if elevation == -70 else 0.55
            camera.lookat[:] = [0.04, -0.03, -0.03] if elevation == -70 else lookat
            self.cameras.append(camera)

        self.root = tk.Tk()
        self.root.title("Shadow Hand 顶部接近阶段三视图回放")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=8, pady=8)
        self.slider = tk.Scale(
            self.root,
            from_=0,
            to=max(0, len(frames) - 1),
            orient=tk.HORIZONTAL,
            length=960,
            showvalue=False,
            command=self.on_slider,
        )
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
        for camera, name in zip(self.cameras, ("OBLIQUE", "TOP", "SIDE")):
            self.renderer.update_scene(self.sim.data, camera=camera)
            view = Image.fromarray(self.renderer.render())
            view = ImageEnhance.Brightness(view).enhance(1.35)
            view = ImageEnhance.Contrast(view).enhance(1.12)
            draw = ImageDraw.Draw(view)
            draw.rectangle((0, 0, 84, 22), fill=(24, 30, 40))
            draw.text((5, 3), name, fill=(255, 255, 255))
            views.append(view)
        image = Image.new("RGB", (960, 280), (224, 228, 235))
        for column, view in enumerate(views):
            image.paste(view, (column * 320, 0))
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)
        self.status.configure(
            text=(
                f"帧 {self.index}/{len(self.frames) - 1}    "
                f"中心误差: {float(info.get('anchor_error', float('nan'))):.4f} m    "
                f"掌面角度误差: {float(info.get('angle_error', float('nan'))) * 180.0 / 3.1415926:.2f} deg    "
                f"最小间隙: {float(info.get('min_gap', float('nan'))):.4f} m    "
                f"顶部接近成功: {info.get('success', False)}"
            )
        )

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
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("rl/checkpoints/shadow_topdown_approach_ppo_v1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    sim, frames, info = collect_rollout(resolve_model_path(args.model), args.seed)
    print(
        f"已记录 {len(frames)} 帧，最终中心误差 {info.get('anchor_error', float('nan')):.4f} m，"
        f"掌面角度误差 {float(info.get('angle_error', float('nan'))) * 180.0 / 3.1415926:.2f} deg，"
        f"成功={info.get('success', False)}"
    )
    ReplayWindow(sim, frames, fps=args.fps).run()


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
