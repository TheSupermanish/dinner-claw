import mujoco
import numpy as np
import pytest

from tabletop_vla.sim.kinematics import gripper_geometry, solve_down
from tabletop_vla.sim.runtime import Simulation


@pytest.mark.parametrize("seed", range(10))
def test_contact_pick_and_place(seed):
    sim = Simulation()
    sim.start_task(seed)
    initial = sim.data.qpos.copy()
    assert sim.model.neq == 0  # No weld or other object attachment.
    for _ in range(20):
        sim.step(500)
        assert np.all(sim.data.ctrl >= sim.model.actuator_ctrlrange[:, 0])
        assert np.all(sim.data.ctrl <= sim.model.actuator_ctrlrange[:, 1])
        if sim.task.status != "running":
            break
    result = sim.task.state()
    assert result["status"] == "succeeded", result
    assert result["peak_lift_m"] > 0.04
    assert result["sustained_lift_s"] > 0.4
    assert result["stable_released_s"] >= 0.3
    assert result["placement_error_m"] < 0.01
    assert not np.array_equal(initial, sim.data.qpos)
    assert not sim.data.warning.number.any()


def test_open_gripper_cannot_pass_lift_gate():
    sim = Simulation()
    sim.start_task(0, close_gripper=False)
    for _ in range(20):
        sim.step(500)
        if sim.task.status != "running":
            break
    assert sim.task.status == "failed"
    assert sim.task.lifted_seconds == 0
    assert "bilateral" in sim.task.reason


def test_ik_is_scratch_only_and_robot_bases_do_not_overlap():
    sim = Simulation()
    initial = sim.data.qpos.copy()
    result = solve_down(sim.model, sim.data, "right", [0.2, -0.06, 0.772])
    assert result.accepted
    np.testing.assert_array_equal(initial, sim.data.qpos)
    np.testing.assert_allclose(sim.model.body("left_base").pos, [-0.23, -0.27, 0.735])
    np.testing.assert_allclose(sim.model.body("right_base").pos, [0.23, -0.27, 0.735])
    assert sim.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC


def test_manual_action_cancels_task_without_claiming_success():
    sim = Simulation()
    sim.start_task(0)
    sim.step(500)
    sim.control("right_gripper", 0.85)
    assert sim.task.status == "cancelled"


@pytest.mark.parametrize("arm", ("left", "right"))
def test_live_gripper_geometry_tracks_asymmetric_jaw(arm):
    sim = Simulation()
    joint = sim.model.joint(f"{arm}_gripper")
    qid = int(joint.qposadr[0])

    sim.data.qpos[qid] = joint.range[1]
    mujoco.mj_forward(sim.model, sim.data)
    opened = gripper_geometry(sim.model, sim.data, arm)

    sim.data.qpos[qid] = joint.range[0]
    mujoco.mj_forward(sim.model, sim.data)
    closed = gripper_geometry(sim.model, sim.data, arm)

    assert np.isfinite(opened.midpoint).all()
    assert np.linalg.norm(opened.closing_axis) == pytest.approx(1.0)
    assert opened.separation > closed.separation
    assert not np.allclose(opened.midpoint, closed.midpoint)
    np.testing.assert_array_equal(sim.data.qpos[qid], joint.range[0])
