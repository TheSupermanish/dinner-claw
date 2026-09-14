from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image

from tabletop_vla.sim.build_scene import build_scene


def render(output: Path, width: int = 960, height: int = 720) -> Path:
    scene = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera="overview")
        pixels = renderer.render()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/scene.png"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    print(render(args.output, args.width, args.height))


if __name__ == "__main__":
    main()
