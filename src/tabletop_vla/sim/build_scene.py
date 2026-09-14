from __future__ import annotations

import copy
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "vendor/so101/so101_new_calib.xml"
OUTPUT = PROJECT / "models/dinner_table.xml"

NAME_ATTRS = {"name"}
REFERENCE_ATTRS = {"joint", "joint1", "joint2", "body", "body1", "body2", "site", "site1", "site2"}


def _namespace_tree(node: ET.Element, prefix: str) -> None:
    for element in node.iter():
        for attr in NAME_ATTRS | REFERENCE_ATTRS:
            value = element.get(attr)
            if value:
                element.set(attr, f"{prefix}{value}")


def _robot_copy(
    root: ET.Element, prefix: str, pos: str, quat: str
) -> tuple[ET.Element, list[ET.Element]]:
    body = copy.deepcopy(root.find("worldbody/body"))
    if body is None:
        raise ValueError("Upstream SO-101 model has no root body")
    actuators = [copy.deepcopy(item) for item in root.findall("actuator/*")]
    # A concave jaw mesh becomes one convex hull in MuJoCo, which can fill
    # visible recesses. Approximate its contact surfaces with convex pieces.
    # Keep upstream visuals/inertia/limits; use explicit convex collision pieces.
    gripper = body.find(".//body[@name='gripper']")
    jaw = body.find(".//body[@name='moving_jaw_so101_v1']")
    for link, mesh in ((gripper, "wrist_roll_follower_so101_v1"),
                       (jaw, "moving_jaw_so101_v1")):
        for geom in link.findall("geom"):
            if geom.get("class") == "collision" and geom.get("mesh") == mesh:
                link.remove(geom)
    for link, name, collision_pos, size in (
        (gripper, "fixed_pad", "-0.0139 -0.000218 -0.0884", "0.006 0.0075 0.016"),
        (gripper, "fixed_neck", "-0.021 0 -0.052", "0.006 0.010 0.024"),
        (gripper, "palm", "0 0 -0.017", "0.030 0.024 0.018"),
        (jaw, "moving_pad", "-0.0058 -0.065 0.0189", "0.0065 0.017 0.0075"),
        (jaw, "moving_neck", "-0.003 -0.030 0.0189", "0.009 0.020 0.010"),
    ):
        ET.SubElement(link, "geom", {
            "name": name, "type": "box", "class": "collision", "pos": collision_pos,
            "size": size, "friction": "1 0.01 0.001", "condim": "4",
            "rgba": "0.2 0.2 0.2 0.35",
        })
    _namespace_tree(body, prefix)
    for actuator in actuators:
        _namespace_tree(actuator, prefix)
    body.set("pos", pos)
    body.set("quat", quat)
    return body, actuators


def build_scene(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    upstream = ET.parse(source).getroot()
    model = ET.Element("mujoco", {"model": "dual_so101_dinner_table"})
    ET.SubElement(
        model,
        "compiler",
        {"angle": "radian", "meshdir": "../vendor/so101/assets", "autolimits": "true"},
    )
    ET.SubElement(model, "option", {
        "timestep": "0.002", "integrator": "implicitfast", "cone": "elliptic",
        "impratio": "10", "solver": "Newton", "noslip_iterations": "3",
    })
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "960"})

    for defaults in upstream.findall("default"):
        model.append(copy.deepcopy(defaults))
    assets = upstream.find("asset")
    if assets is not None:
        model.append(copy.deepcopy(assets))

    world = ET.SubElement(model, "worldbody")
    ET.SubElement(
        world, "light", {"name": "key", "pos": "0 -0.3 2.2", "dir": "0 0 -1", "directional": "true"}
    )
    ET.SubElement(world, "body", {"name": "lookat", "pos": "0 0 0.94"})
    camera_pos = np.array([0.95, -1.2, 1.65])
    z_axis = camera_pos - [0, 0, 0.85]
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0, 0, 1], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    ET.SubElement(
        world,
        "camera",
        {
            "name": "overview",
            "pos": " ".join(map(str, camera_pos)),
            "xyaxes": " ".join(map(str, np.r_[x_axis, y_axis])),
            "fovy": "48",
        },
    )
    ET.SubElement(
        world, "camera", {"name": "top", "pos": "0 0 2.45", "quat": "1 0 0 0", "fovy": "48"}
    )
    ET.SubElement(world, "light", {"pos": "-1 -1 2", "dir": "0 0 -1", "diffuse": "0.5 0.5 0.5"})
    ET.SubElement(
        world,
        "geom",
        {"name": "floor", "type": "plane", "size": "2 2 0.05", "rgba": "0.12 0.14 0.17 1"},
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "table",
            "type": "box",
            "pos": "0 0 0.70",
            "size": "0.62 0.48 0.035",
            "rgba": "0.55 0.36 0.20 1",
            "friction": "1 0.01 0.001",
        },
    )

    left, left_actuators = _robot_copy(
        upstream, "left_", "-0.23 -0.27 0.735", "0.7071068 0 0 0.7071068"
    )
    right, right_actuators = _robot_copy(
        upstream, "right_", "0.23 -0.27 0.735", "0.7071068 0 0 0.7071068"
    )
    world.extend([left, right])

    for name, pos, size, rgba in (
        ("plate", "-0.12 0.02 0.752", "0.10 0.004", "0.92 0.92 0.88 1"),
        ("mug", "0.20 -0.06 0.7603", "0.025 0.004", "0.18 0.48 0.86 1"),
        ("bottle", "0.28 0.12 0.84", "0.035 0.10", "0.25 0.72 0.52 1"),
    ):
        body = ET.SubElement(world, "body", {"name": name, "pos": pos})
        ET.SubElement(body, "freejoint")
        geom_type = "cylinder"
        geom = ET.SubElement(
            body,
            "geom",
            {"name": f"{name}_geom", "type": geom_type, "size": size, "rgba": rgba, "mass": "0.12"},
        )
        if name == "plate":
            # A real dinner plate has a rim standing proud of a recessed base; the flat
            # disc was the simplification. Measured 2026-09-14: the gripper hangs 50 mm
            # below its own jaw midpoint, so a rim flush with the table (z 0.735-0.751)
            # cannot be pinched. The probe drove 63-177 N into the plate and 39 N into
            # the table, shoved it 19-149 mm, and never lifted it (0/10). This rim gives
            # a 20 mm grip band starting 8 mm above the table, the same profile as the
            # cutlery handle that does work.
            geom.set("pos", "0 0 0")
            geom.set("mass", "0.06")
            for k in range(12):
                angle = 2 * np.pi * k / 12
                ET.SubElement(body, "geom", {
                    "name": f"plate_rim_{k}",
                    "type": "box",
                    "pos": f"{0.094 * np.cos(angle)} {0.094 * np.sin(angle)} 0.014",
                    "quat": f"{np.cos(angle / 2)} 0 0 {np.sin(angle / 2)}",
                    "size": "0.010 0.025 0.010",
                    "mass": "0.005",
                    "rgba": rgba,
                    "condim": "4",
                    "friction": "1 0.01 0.001",
                })
        if name == "mug":
            # A miniature square hollow cup: 50 mm width/height, 60 g total mass.
            # Separate convex walls preserve the cavity in MuJoCo contact physics.
            geom.set("pos", "0 0 -0.021")
            geom.set("type", "box")
            geom.set("size", "0.025 0.025 0.004")
            geom.set("mass", "0.012")
            for k in range(4):
                angle = 2 * np.pi * k / 4
                ET.SubElement(body, "geom", {
                    "name": f"mug_wall_{k}", "type": "box",
                    "pos": f"{0.022 * np.cos(angle)} {0.022 * np.sin(angle)} 0.004",
                    "quat": f"{np.cos(angle / 2)} 0 0 {np.sin(angle / 2)}",
                    "size": "0.003 0.025 0.021", "mass": "0.012", "rgba": rgba,
                    "condim": "4", "friction": "1 0.01 0.001",
                })

    actuator = ET.SubElement(model, "actuator")
    actuator.extend(left_actuators + right_actuators)
    drawer = ET.SubElement(world, "body", {"name": "drawer", "pos": "0 0.25 0.752"})
    ET.SubElement(
        drawer,
        "joint",
        {
            "name": "drawer_slide",
            "type": "slide",
            "axis": "0 -1 0",
            "range": "0 0.12",
            "damping": "4",
        },
    )
    for name, pos, size in (
        ("floor", "0 0 0", "0.13 0.075 0.009"),
        ("front", "0 -0.075 0.025", "0.13 0.008 0.025"),
        ("back", "0 0.075 0.025", "0.13 0.008 0.025"),
        ("left", "-0.13 0 0.025", "0.008 0.075 0.025"),
        ("right", "0.13 0 0.025", "0.008 0.075 0.025"),
        ("handle", "0 -0.1 0.025", "0.045 0.008 0.008"),
    ):
        ET.SubElement(
            drawer,
            "geom",
            {
                "name": f"drawer_{name}",
                "type": "box",
                "pos": pos,
                "size": size,
                "mass": "0.06",
                "rgba": "0.27 0.34 0.4 1",
            },
        )
    for name in ("left", "right"):
        ET.SubElement(drawer, "geom", {"name": f"drawer_mount_{name}", "type": "box",
                                       "pos": "0 0 -3", "size": "0.004 0.041 0.006",
                                       "mass": "0", "contype": "0", "conaffinity": "0",
                                       "rgba": "0.27 0.34 0.4 1"})
    for obj, x in (("fork", -0.045), ("spoon", 0.045)):
        body = ET.SubElement(world, "body", {"name": obj, "pos": f"{x} 0.25 0.766"})
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obj}_geom",
                "type": "box",
                "size": "0.008 0.05 0.003",
                "mass": "0.015",
                "rgba": "0.72 0.78 0.84 1",
            },
        )
        # A raised handle, the way real cutlery has one. Measured 2026-09-14: the bare
        # 6 mm blade lying on the drawer floor cannot be grasped, because the 15-32 mm
        # jaw pads would have to penetrate the floor to straddle it. The jaws skidded
        # the blade sideways at 7.6 N instead of lifting it. The handle sits on top of
        # the blade at one end and gives the jaws 16 mm of height clear of the floor.
        # Same precedent as the cup, which also needed a graspable compound geometry.
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obj}_handle",
                "type": "box",
                # Measured 2026-09-14: a 36 mm handle cannot host two grippers. The two
                # moving pads (13 x 34 x 15 mm each) overlapped by 9.7 mm and needed
                # 44-50 mm of handle between them, so the bimanual handoff was impossible.
                # 56 mm, centred so the far end stays on the 100 mm blade.
                "pos": "0 -0.022 0.011",
                "size": "0.007 0.028 0.008",
                "mass": "0.010",
                "rgba": "0.72 0.78 0.84 1",
                "condim": "4",
                "friction": "1 0.01 0.001",
            },
        )
    # Dormant housing geometry is activated only by the drawer workcell reset.
    # Keep the learned cup scene's rendered appearance unchanged.
    for name, size in (("left", "0.008 0.087 0.048"),
                       ("right", "0.008 0.087 0.048"),
                       ("back", "0.155 0.008 0.048"),
                       ("top", "0.155 0.087 0.008")):
        ET.SubElement(world, "geom", {"name": f"cabinet_{name}", "type": "box",
                                     "pos": "0 0 -3", "size": size,
                                     "rgba": "0.22 0.26 0.30 1"})
    for obj, x, y in (
        ("plate", -0.2, 0.12),
        ("mug", 0.27, -0.04),
        ("fork", -0.32, 0.1),
        ("spoon", 0.32, 0.1),
    ):
        ET.SubElement(
            world,
            "site",
            {
                "name": f"{obj}_mat",
                "type": "box",
                "pos": f"{x} {y} 0.737",
                "size": "0.045 0.055 0.001",
                "rgba": "0.1 0.8 0.65 0.3",
            },
        )
    ET.SubElement(model, "sensor")

    ET.indent(model, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Several evaluation/viewer processes may build the same scene concurrently.
    # Publish a complete document atomically so readers never see a partial XML.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".xml", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            ET.ElementTree(model).write(handle, encoding="utf-8", xml_declaration=True)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    print(build_scene())


if __name__ == "__main__":
    main()
