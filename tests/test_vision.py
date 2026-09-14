import numpy as np
import pytest

from tabletop_vla.perception.cup import locate_blue_cup
from tabletop_vla.planning.task_planner import executable_task
from tabletop_vla.sim.runtime import Simulation


def test_grounding_moves_with_pixels_and_rejects_blank_image():
    image = np.zeros((600, 900, 3), dtype=np.uint8)
    image[314:335, 520:541] = [50, 140, 220]
    first = locate_blue_cup(image)
    second = locate_blue_cup(np.roll(image, 10, axis=1))
    assert second["xy"][0] - first["xy"][0] > 0.02
    with pytest.raises(ValueError, match="visible"):
        locate_blue_cup(np.zeros_like(image))


def test_camera_task_cannot_use_hidden_pose_when_image_is_missing(monkeypatch):
    sim = Simulation()
    monkeypatch.setattr(sim, "rgb", lambda camera: np.zeros((600, 900, 3), dtype=np.uint8))
    sim.start_task(0, "camera-cup-place")
    sim.step(500)
    assert sim.task.status == "failed"
    assert "camera image" in sim.task.reason


@pytest.mark.parametrize("instruction", [
    "Do not pick up the cup", "Pick up the cup with the left arm and place it on its marker",
    "Pick up the blue cup and place it on its marker, then open the drawer",
    "Pick up the plate and place it on its marker",
])
def test_unsupported_instruction_cannot_partially_execute(instruction):
    with pytest.raises(ValueError, match="Only the verified"):
        executable_task(instruction)


def test_supported_instruction_uses_camera_task():
    assert executable_task("Pick up the blue cup and place it on its marker.") == "camera-cup-place"


@pytest.mark.parametrize("seed", [310, 311, 312])
def test_real_pixels_drive_verified_pickup(seed):
    sim = Simulation()
    try:
        sim.start_task(seed, "camera-cup-place")
        for _ in range(20):
            sim.step(500)
            if sim.task.status != "running":
                break
        assert sim.task.status == "succeeded", sim.task.state()
        assert sim.task.observation["visible_pixels"] >= 30
    finally:
        sim.close()


def test_missed_camera_grasp_retries_without_reset(monkeypatch):
    sim = Simulation()
    try:
        sim.start_task(534, "camera-cup-place")
        def unexpected_reset(seed):
            pytest.fail("Recovery must not reset the randomized scene")
        monkeypatch.setattr(sim, "reset", unexpected_reset)
        for _ in range(40):
            sim.step(500)
            if sim.task.status != "running":
                break
        result = sim.task.state()
        assert result["status"] == "succeeded", result
        assert result["attempt"] == 2
    finally:
        sim.close()
