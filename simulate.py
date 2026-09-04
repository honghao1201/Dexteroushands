"""启动 Aero Hand Open 的 MuJoCo 可视化仿真。"""

from __future__ import annotations

import argparse  # 解析命令行参数。
import pathlib  # 处理跨平台文件路径。
import time  # 控制仿真循环的实时速度。

import mujoco  # MuJoCo Python 接口。
import mujoco.viewer  # MuJoCo 被动查看器。


# 项目根目录和默认的右手场景文件。
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_SCENE = PROJECT_ROOT / "models" / "tetheria_aero_hand_open" / "scene_right.xml"


def main() -> None:
    """加载 MJCF 场景，并在窗口中持续推进物理仿真。"""
    # 定义可选的场景路径，未指定时使用项目内置场景。
    parser = argparse.ArgumentParser(description="Run the Aero Hand Open MuJoCo viewer")
    parser.add_argument("--scene", type=pathlib.Path, default=DEFAULT_SCENE)
    args = parser.parse_args()

    # 解析路径并在加载前给出清晰的文件错误。
    scene = args.scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene}")

    # 从 XML 创建模型和运行时数据对象。
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    # 输出模型规模，便于确认关节、腱绳和执行器是否正确加载。
    print(f"Loaded: {scene}")
    print(f"Joints: {model.njnt} | Tendons: {model.ntendon} | Actuators: {model.nu}")

    # 启动被动查看器；窗口关闭后退出循环。
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 记录本次物理步开始时间，用于按 MuJoCo 时间步限速。
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            # 如果计算得很快，则等待剩余时间，使画面接近实时速度。
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
