import numpy as np
import pytest

from tabletop_vla.sim.plate import (
    ALIGNED_PHASES,
    MAT,
    MAT_PHASES,
    PLATE_JITTER,
    PLATE_START,
    PlatePlace,
    prepare_plate_workcell,
)
from tabletop_vla.sim.runtime import Simulation


def run(seed, close_gripper=True, batches=60):
    sim = Simulation()
    sim.reset(seed)
    prepare_plate_workcell(sim, seed)
    sim.task = PlatePlace(sim, close_gripper=close_gripper)
    sim.running = True
    for _ in range(batches):
        sim.step(500)
        if sim.task.status != "running":
            break
    return sim, sim.task.state()


def test_workcell_parks_plate_before_episode_without_touching_geometry():
    sim = Simulation()
    try:
        sim.reset(40)
        baseline_geom = sim.model.geom_pos.copy()
        baseline_body = sim.model.body_pos.copy()
        prepare_plate_workcell(sim, 40)
        plate = sim.data.xpos[sim.model.body("plate").id].copy()
        np.testing.assert_allclose(plate[:2], PLATE_START[:2], atol=PLATE_JITTER + 1e-9)
        assert plate[2] == pytest.approx(0.752)
        # The mat target and the scene geometry are untouched by the workcell.
        np.testing.assert_allclose(
            sim.data.site("plate_mat").xpos, [-0.20, 0.12, 0.737], atol=1e-6)
        np.testing.assert_array_equal(sim.model.geom_pos, baseline_geom)
        np.testing.assert_array_equal(sim.model.body_pos, baseline_body)
        assert sim.variation["plate_workcell"]["plate_start"][:2] == pytest.approx(
            plate[:2].tolist())
    finally:
        sim.close()


def test_runtime_constructs_plate_task_and_still_rejects_bare_plate_place():
    """`plate-place` stays rejected: tests/test_cutlery.py:35 reserves that exact
    string, so this task is whitelisted as `plate-place-left` instead."""
    sim = Simulation()
    try:
        sim.start_task(40, "plate-place-left")
        assert isinstance(sim.task, PlatePlace)
        assert sim.task.status == "running"
        with pytest.raises(ValueError):
            sim.start_task(40, "plate-place")
    finally:
        sim.close()


def test_phase_sets_name_real_phases():
    sim = Simulation()
    try:
        sim.reset(40)
        prepare_plate_workcell(sim, 40)
        names = {name for name, _ in PlatePlace(sim).phases}
        assert MAT_PHASES <= names
        assert ALIGNED_PHASES <= names
    finally:
        sim.close()


def test_plate_is_grasped_and_lifted_but_the_carry_is_not_solved():
    """Measured 2026-09-14, outputs/plate-restored-40-49.json.

    The original assertions here encoded the pre-rim reality: a flat 200 mm disc flush
    with the table could not be pinched at all, because the gripper hangs 50 mm below its
    own jaw midpoint. The probe drove 39 N into the table and never lifted it, 0/10.

    Modelling the plate the way a real plate is built, a base disc with a raised rim,
    fixed the grasp outright: all ten seeds now hold and lift it for 1.06-3.26 s against
    a 0.4 s gate. The task still scores 0/10 because the carry waypoint
    "Raise to transit height" finds no accepted approach angle. That is recorded here
    rather than asserted away, and the lift assertion below is what stops the grasp
    silently regressing while the carry is being fixed.
    """
    sim, result = run(40)
    try:
        assert result["sustained_lift_s"] >= 0.4, result
        assert sim.task.bilateral_s >= 0.15, result
        # Honest: the full task does not yet succeed, and the reason is the carry.
        assert result["status"] == "failed", result
        assert "Unreachable" in (result["reason"] or ""), result
    finally:
        sim.close()


def test_open_gripper_control_cannot_reach_the_grasp_gate():
    """A gate that cannot fail is not a gate."""
    sim, result = run(40, close_gripper=False)
    try:
        assert result["status"] == "failed"
        assert result["bilateral_contact_s"] == 0.0
        assert result["sustained_lift_s"] == 0.0
    finally:
        sim.close()


def test_mat_is_distinct_from_the_plate_start():
    assert np.linalg.norm(MAT[:2] - PLATE_START[:2]) > 0.05
