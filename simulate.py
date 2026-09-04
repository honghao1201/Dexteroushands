"""Launch the Aero Hand Open MuJoCo scene."""

from __future__ import annotations

import argparse
import pathlib
import time

import mujoco
import mujoco.viewer


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_SCENE = PROJECT_ROOT / "models" / "tetheria_aero_hand_open" / "scene_right.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Aero Hand Open MuJoCo viewer")
    parser.add_argument("--scene", type=pathlib.Path, default=DEFAULT_SCENE)
    args = parser.parse_args()

    scene = args.scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene}")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    print(f"Loaded: {scene}")
    print(f"Joints: {model.njnt} | Tendons: {model.ntendon} | Actuators: {model.nu}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
