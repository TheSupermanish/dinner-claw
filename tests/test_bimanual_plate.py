import mujoco
import numpy as np
import pytest

from tabletop_vla.sim.bimanual_plate import (
    ARMS,
    LEVEL_LIMIT_DEG,
    PLATE_JITTER,
    PLATE_MAT,
    PLATE_START,
    BimanualPlateLift,
    prepare_bimanual_workcell,
)
from tabletop_vla.sim.runtime import Simulation


def run(seed, close_gripper=True, batches=60):
    sim = Simulation()
    sim.reset(seed)
    prepare_bimanual_workcell(sim, seed)
    sim.task = BimanualPlateLift(sim, close_gripper=close_gripper)
    sim.running = True
    for _ in range(batches):
        sim.step(500)
        if sim.task.status != "running":
            break
    return sim, sim.task.state()


def test_workcell_parks_plate_and_mat_before_episode_without_touching_geometry():
    sim = Simulation()
    try:
        sim.reset(41)
        baseline_geom = sim.model.geom_pos.copy()
        baseline_body = sim.model.body_pos.copy()
        prepare_bimanual_workcell(sim, 41)
        plate = sim.data.xpos[sim.model.body("plate").id].copy()
        np.testing.assert_allclose(plate[:2], PLATE_START[:2], atol=PLATE_JITTER + 1e-9)
        # Both ends of the carry must sit in the two-arm overlap. The authored mat and
        # the single-arm workcell's mat at x=-0.14 are both outside it, so the move is
        # deliberate and has to be declared in the episode result.
        np.testing.assert_allclose(sim.data.site("plate_mat").xpos, PLATE_MAT, atol=1e-6)
        declared = sim.variation["bimanual_plate_workcell"]
        assert declared["plate_mat_moved_into_two_arm_overlap"] == PLATE_MAT.tolist()
        assert declared["plate_start"][:2] == pytest.approx(plate[:2].tolist())
        # Parking an object is allowed; editing the scene is not.
        np.testing.assert_array_equal(sim.model.geom_pos, baseline_geom)
        np.testing.assert_array_equal(sim.model.body_pos, baseline_body)
    finally:
        sim.close()


def test_open_gripper_control_never_scores_a_lift():
    """The negative control is the check that the gates measure the robot's work.

    With both grippers held open the plate must never register a sustained lift, no
    matter what the arms do above it.
    """
    sim, state = run(41, close_gripper=False)
    try:
        assert state["status"] == "failed"
        assert state["sustained_lift_s"] == 0.0
        assert state["stable_release_s"] == 0.0
    finally:
        sim.close()


def test_lift_requires_all_four_pads_not_one_arms_two():
    """Two pads bearing force is a single-arm grasp, and a single arm levers the plate
    instead of lifting it: measured 2026-09-15, six of ten single-arm seeds reached
    38-55 mm of centre height with 0.000-0.008 s of sustained lift because the plate
    never left the table. The gate counts all four or nothing."""
    sim = Simulation()
    try:
        sim.reset(41)
        prepare_bimanual_workcell(sim, 41)
        task = BimanualPlateLift(sim)
        assert len(task.all_pads) == 4
        for arm in ARMS:
            assert len(task.pads[arm]) == 2
            # One arm alone can never satisfy the contact gate.
            assert not task.all_pads <= task.pads[arm]
    finally:
        sim.close()


def test_a_tipped_plate_cannot_score_a_lift_however_high_its_centre_rises():
    """The levelness bound is what separates lifting from levering.

    A plate pivoting on its far rim raises its centre while still resting on the
    table, which passed a height-only gate. Drive the plate to a steep tilt and
    confirm the lift accumulator refuses it.
    """
    sim = Simulation()
    try:
        sim.reset(41)
        prepare_bimanual_workcell(sim, 41)
        task = BimanualPlateLift(sim)
        body = sim.model.body("plate").id
        address = sim.model.jnt_qposadr[sim.model.body_jntadr[body]]
        # Well clear of the table in height, and tipped far past the bound.
        sim.data.qpos[address:address + 3] = [PLATE_START[0], PLATE_START[1], 0.95]
        half = np.deg2rad(40.0) / 2
        sim.data.qpos[address + 3:address + 7] = [np.cos(half), np.sin(half), 0.0, 0.0]
        mujoco.mj_forward(sim.model, sim.data)
        assert task.tilt_deg() == pytest.approx(40.0, abs=0.5)
        assert task.tilt_deg() > LEVEL_LIMIT_DEG
        assert task.lift > 0.025
        task.after_step()
        # Height is there and the tilt is not, so nothing accumulates.
        assert task.sustained_lift == 0.0
        assert task.stable_release == 0.0
    finally:
        sim.close()


def test_retract_aims_from_the_live_jaw_position_not_the_nominal_mat():
    """Two single-arm retract rewrites failed by aiming at `mat + fixed offset`, which
    commands a sideways move equal to however far the plate drifted. The retract target
    has to follow the plate."""
    sim = Simulation()
    try:
        sim.reset(41)
        prepare_bimanual_workcell(sim, 41)
        task = BimanualPlateLift(sim)
        body = sim.model.body("plate").id
        address = sim.model.jnt_qposadr[sim.model.body_jntadr[body]]
        sim.data.qpos[address:address + 3] = [PLATE_MAT[0], PLATE_MAT[1], 0.752]
        mujoco.mj_forward(sim.model, sim.data)
        on_mat = task.target_for("Retract")
        # Now shift the plate 40 mm and confirm the retract target moves with it.
        sim.data.qpos[address] = PLATE_MAT[0] + 0.04
        mujoco.mj_forward(sim.model, sim.data)
        drifted = task.target_for("Retract")
        for arm in ARMS:
            assert drifted[arm][0] - on_mat[arm][0] == pytest.approx(0.04, abs=1e-6)
            assert drifted[arm][2] == pytest.approx(on_mat[arm][2], abs=1e-9)
    finally:
        sim.close()
