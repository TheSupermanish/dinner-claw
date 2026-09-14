from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from tabletop_vla.sim.build_scene import OUTPUT, build_scene


def run(steps: int = 250) -> dict[str, object]:
    scene = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return {
        "scene": str(Path(OUTPUT)),
        "bodies": model.nbody,
        "joints": model.njnt,
        "actuators": model.nu,
        "finite_state": bool(np.isfinite(data.qpos).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    print(run(args.steps))


if __name__ == "__main__":
    main()
