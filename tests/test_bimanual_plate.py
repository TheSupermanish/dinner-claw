import mujoco
import numpy as np
import pytest

import tabletop_vla.sim.bimanual_plate as plate_mod
from tabletop_vla.sim.bimanual_plate import (
    ARMS,
    LEVEL_LIMIT_DEG,
    PLATE_JITTER,
    PLATE_MAT,
    PLATE_START,
    RIM_R,
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


def _plate_address(sim):
    body = sim.model.body("plate").id
    return body, sim.model.jnt_qposadr[sim.model.body_jntadr[body]]


def _set_plate_pose(sim, address, xyz, tilt_deg):
    sim.data.qpos[address:address + 3] = list(xyz)
    half = np.deg2rad(float(tilt_deg)) / 2
    sim.data.qpos[address + 3:address + 7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    mujoco.mj_forward(sim.model, sim.data)


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

    With both grippers held open the plate must fail at the close-phase
    four-pad gate, not later. Checking only the final failed status lets a
    build with the close gate disabled still pass, because the episode then
    fails one phase later at the lift gate with the same failed status.
    """
    sim, state = run(41, close_gripper=False)
    try:
        assert state["status"] == "failed"
        assert state["sustained_lift_s"] == 0.0
        assert state["stable_release_s"] == 0.0
        # The close gate is the production line under test: without it this
        # control proceeds to "Lift clear of table" and fails there instead.
        assert state["phase"] == "Close on rims"
        assert "four-pad grasp" in state["reason"]
        assert state["four_pad_contact_s"] == 0.0
    finally:
        sim.close()


def test_lift_requires_all_four_pads_not_one_arms_two():
    """Two pads bearing force is a single-arm grasp, and a single arm levers the plate
    instead of lifting it: measured 2026-09-15, six of ten single-arm seeds reached
    38-55 mm of centre height with 0.000-0.008 s of sustained lift because the plate
    never left the table. The gate counts all four or nothing.

    This test drives the production gate in `after_step`, not the set objects.
    A build that accepts ANY pad instead of ALL must accumulate lift here.
    """
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
        _, address = _plate_address(sim)
        start = task.start.copy()
        # Level, airborne, and otherwise liftable: tilt well inside the bound and
        # height well above the 25 mm airborne line, so contact is the only variable.
        _set_plate_pose(sim, address, [start[0], start[1], start[2] + 0.05], 5.0)
        assert task.tilt_deg() == pytest.approx(5.0, abs=0.5)
        assert task.tilt_deg() < LEVEL_LIMIT_DEG
        assert task.lift > 0.025
        # One arm's two pads: the production gate must refuse all accumulation.
        task.contacts = lambda: set(task.pads["left"])
        for _ in range(10):
            task.after_step()
        assert task.quad_s == 0.0
        assert task.sustained_lift == 0.0
        assert task.peak_tilt_deg == 0.0
        # Positive control on the same fixture: all four pads must accumulate,
        # proving the refusal above came from the contact gate and not a dead fixture.
        task.current_contact = 0.0
        task.quad_s = 0.0
        task.lift_run = 0.0
        task.sustained_lift = 0.0
        task.peak_tilt_deg = 0.0
        task.tilt_at_peak_lift_deg = 0.0
        task.contacts = lambda: set(task.all_pads)
        for _ in range(10):
            task.after_step()
        assert task.quad_s > 0.0
        assert task.sustained_lift > 0.0
        assert task.peak_tilt_deg == pytest.approx(5.0, abs=0.5)
    finally:
        sim.close()


def test_a_tipped_plate_cannot_score_a_lift_however_high_its_centre_rises():
    """The levelness bound is what separates lifting from levering.

    A plate pivoting on its far rim raises its centre while still resting on the
    table, which passed a height-only gate. This fixture holds every other lift
    condition true (four-pad contact, whitelist support, airborne height) and lets
    tilt ALONE refuse the lift. A build with the tilt condition removed must
    accumulate sustained lift here.
    """
    sim = Simulation()
    try:
        sim.reset(41)
        prepare_bimanual_workcell(sim, 41)
        task = BimanualPlateLift(sim)
        _, address = _plate_address(sim)
        start = task.start.copy()
        # Tipped far past the bound but well clear of the table in height.
        _set_plate_pose(sim, address, [start[0], start[1], start[2] + 0.10], 40.0)
        assert task.tilt_deg() == pytest.approx(40.0, abs=0.5)
        assert task.tilt_deg() > LEVEL_LIMIT_DEG
        assert task.lift > 0.025
        # Every non-tilt condition holds: all four pads, nothing else touching.
        task.contacts = lambda: set(task.all_pads)
        for _ in range(25):
            task.after_step()
        # Held, so contact accumulated; airborne, so the peak tilt recorded the
        # tip; level-gated, so no sustained lift followed.
        assert task.quad_s > 0.0
        assert task.peak_tilt_deg == pytest.approx(40.0, abs=0.5)
        assert task.sustained_lift == 0.0
        assert task.stable_release == 0.0
        # Positive control with the same contacts and height: level the plate and
        # the same gate must accumulate, proving the refusal was tilt alone.
        task.current_contact = 0.0
        task.quad_s = 0.0
        task.lift_run = 0.0
        task.sustained_lift = 0.0
        task.peak_tilt_deg = 0.0
        task.tilt_at_peak_lift_deg = 0.0
        _set_plate_pose(sim, address, [start[0], start[1], start[2] + 0.10], 5.0)
        assert task.tilt_deg() < LEVEL_LIMIT_DEG
        assert task.lift > 0.025
        for _ in range(10):
            task.after_step()
        assert task.sustained_lift > 0.0
        assert task.peak_tilt_deg == pytest.approx(5.0, abs=0.5)
    finally:
        sim.close()


def test_retract_aims_from_the_live_jaw_position_not_the_nominal_mat(monkeypatch):
    """Two single-arm retract rewrites failed by aiming at `mat + fixed offset`, which
    commands a sideways move equal to however far the plate drifted. The retract target
    has to follow the plate, on the path `enter()` actually takes.

    The helper half guards `retract_candidates()`; the `enter()` half guards the
    dispatch inside production `enter()`, which can bypass the helper. A build where
    `enter()` aims at the nominal mat must fail the second half.
    """
    sim = Simulation()
    try:
        sim.reset(41)
        prepare_bimanual_workcell(sim, 41)
        task = BimanualPlateLift(sim)
        _, address = _plate_address(sim)
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

        def capture_enter_targets():
            captured = {}

            def fake_waypoint(model, data, arm, target, preferred, initial):
                captured[arm] = np.array(target, dtype=float)

                class _Accepted:
                    accepted = True
                    targets = np.zeros(5)
                    position_error = 0.0
                    axis_error = 0.0

                return _Accepted()

            monkeypatch.setattr(plate_mod, "_waypoint", fake_waypoint)
            task.enter(8)
            assert set(captured) == set(ARMS)
            return {arm: captured[arm].copy() for arm in ARMS}

        sim.data.qpos[address:address + 3] = [PLATE_MAT[0], PLATE_MAT[1], 0.752]
        mujoco.mj_forward(sim.model, sim.data)
        on_mat_enter = capture_enter_targets()
        # Absolute check against the live plate, not the nominal mat alone: with the
        # plate parked on the mat the live x equals the mat x, so the jaw targets
        # sit one rim radius to each side of it.
        assert on_mat_enter["left"][0] == pytest.approx(PLATE_MAT[0] - RIM_R, abs=1e-9)
        assert on_mat_enter["right"][0] == pytest.approx(PLATE_MAT[0] + RIM_R, abs=1e-9)
        sim.data.qpos[address] = PLATE_MAT[0] + 0.04
        mujoco.mj_forward(sim.model, sim.data)
        drifted_enter = capture_enter_targets()
        for arm in ARMS:
            assert drifted_enter[arm][0] - on_mat_enter[arm][0] == pytest.approx(
                0.04, abs=1e-6)
            assert drifted_enter[arm][2] == pytest.approx(
                on_mat_enter[arm][2], abs=1e-9)
    finally:
        sim.close()
