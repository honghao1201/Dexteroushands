"""导出真实环境复位后的初始图，方便检查球与掌面的水平间隙。"""

from __future__ import annotations

import argparse
import pathlib

import mujoco
from PIL import Image, ImageDraw, ImageFont

from aero_grasp_env import AeroGraspEnv, ROOT


def main() -> None:
    """加载与训练相同的初始状态，分别从斜上方和正上方截图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "outputs" / "initial_vertical_state.png")
    args = parser.parse_args()
    env = AeroGraspEnv()
    env.reset(seed=args.seed)

    # 使用几何距离检查全部手部外形，包括没有参与碰撞的可视网格。
    hand_geoms = [g for g in range(env.model.ngeom) if env.model.geom_bodyid[g] not in (0, env.object_body)]
    clearance = min(
        mujoco.mj_geomDistance(env.model, env.data, env.object_geom, g, 0.5, None)
        for g in hand_geoms
    )
    ball = env.data.xpos[env.object_body].copy()

    # 图片由 MuJoCo 直接渲染；不推进物理、不移动模型、不改变球体高度。
    output = Image.new("RGB", (1280, 550), "white")
    draw = ImageDraw.Draw(output)
    font_path = pathlib.Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
    views = [(315, -20, "斜视：球已水平移到手掌外侧"), (270, -90, "俯视：检查水平面内的分离间隙")]
    with mujoco.Renderer(env.model, height=480, width=640) as renderer:
        for i, (azimuth, elevation, title) in enumerate(views):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [0.15, -0.055, -0.015]
            camera.distance = 0.40
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(env.data, camera)
            output.paste(Image.fromarray(renderer.render()), (640 * i, 34))
            draw.text((640 * i + 12, 4), title, font=font, fill="black")

    # 坐标和间隙均来自本次真实复位状态，方便与训练中的随机位置核对。
    draw.text((12, 520), f"球心 (m)：X={ball[0]:.4f}，Y={ball[1]:.4f}，Z={ball[2]:.4f}（高度不变）；最近外形间隙：{clearance * 1000:.1f} mm", font=font, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    env.close()
    print(f"Saved: {args.output.resolve()}")
    print(f"Initial ball position: {ball}; minimum hand geometry clearance: {clearance * 1000:.3f} mm")


if __name__ == "__main__":
    main()
