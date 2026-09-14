import xml.etree.ElementTree as ET

from tabletop_vla.sim.build_scene import build_scene


def test_scene_has_two_namespaced_robots():
    root = ET.parse(build_scene()).getroot()
    names = {element.get("name") for element in root.iter() if element.get("name")}
    assert "left_shoulder_pan" in names
    assert "right_shoulder_pan" in names
    assert "plate" in names
    assert len(root.findall("actuator/position")) == 12
