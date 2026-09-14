import numpy as np
import pytest

from tabletop_vla.sim.runtime import Simulation


@pytest.fixture
def sim():
    simulation = Simulation()
    yield simulation
    simulation.close()


def test_seed_reproducibility_and_variation(sim):
    first = sim.reset(7)["objects"]
    masses = sim.model.body_mass.copy()
    sim.step(150)
    assert sim.reset(7)["objects"] == first
    np.testing.assert_array_equal(sim.model.body_mass, masses)
    second = sim.reset(8)["objects"]
    assert first != second
    assert all(first[name] != second[name] for name in first)


def test_invalid_learned_start_preserves_existing_scene(sim):
    sim.reset(7)
    before = sim.data.qpos.copy()
    sim.learned_available = False
    with pytest.raises(ValueError, match="train and export"):
        sim.start_task(8, "learned-cup-place")
    assert sim.seed == 7
    np.testing.assert_array_equal(sim.data.qpos, before)
    with pytest.raises(ValueError, match="Seed"):
        sim.reset(True)


def test_ten_seed_physics_stability(sim):
    for seed in range(10):
        sim.reset(seed)
        sim.step(500)
        assert np.isfinite(sim.data.qpos).all()
        assert all(sim.data.body(obj).xpos[2] > 0.7 for obj in ("plate", "mug", "fork", "spoon"))
        assert sim.data.warning.number.sum() == 0


def test_controls_move_through_physics_without_teleporting(sim):
    initial = sim.data.qpos.copy()
    sim.control("left_shoulder_pan", 0.15)
    np.testing.assert_array_equal(sim.data.qpos, initial)
    sim.step(500)
    joint = sim.model.joint("left_shoulder_pan")
    assert sim.data.qpos[joint.qposadr[0]] > 0.05
    with pytest.raises(ValueError):
        sim.control("left_shoulder_pan", 99)


def test_ik_does_not_teleport_and_rejects_invalid_target(sim):
    before = sim.data.qpos.copy()
    target = sim.data.site("left_gripperframe").xpos + [0, 0, 0.03]
    result = sim.reach("left", target)
    assert result["last_motion"]["accepted"]
    np.testing.assert_array_equal(sim.data.qpos, before)
    initial_error = result["last_motion"]["actual_error_m"]
    result = sim.step(500)
    assert result["last_motion"]["actual_error_m"] < initial_error
    with pytest.raises(ValueError):
        sim.reach("left", [float("nan"), 0, 1])
