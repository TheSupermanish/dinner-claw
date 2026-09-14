import numpy as np
import pytest

from tabletop_vla.sim.cutlery import ALIGNED_PHASES, MAT_PHASES, MATS, CutleryRetrieve
from tabletop_vla.sim.drawer import prepare_workcell
from tabletop_vla.sim.runtime import Simulation


def run(seed, obj="fork", close_gripper=True, batches=70):
    sim = Simulation()
    sim.reset(seed)
    prepare_workcell(sim, seed)
    sim.task = CutleryRetrieve(sim, obj, close_gripper=close_gripper)
    sim.running = True
    for _ in range(batches):
        sim.step(500)
        if sim.task.status != "running":
            break
    return sim, sim.task.state()


def test_only_implemented_cutlery_is_accepted():
    sim = Simulation()
    try:
        with pytest.raises(ValueError):
            CutleryRetrieve(sim, "bottle")
    finally:
        sim.close()


def test_runtime_rejects_an_unimplemented_task_name():
    sim = Simulation()
    try:
        with pytest.raises(ValueError):
            sim.start_task(30, "plate-place")
    finally:
        sim.close()


def test_phase_sets_name_real_phases():
    """These sets were substring matches once, and 'Release' silently fell through to the
    wrong wrist azimuth. Every name in them must exist in the phase table."""
    sim = Simulation()
    try:
        sim.reset(30)
        prepare_workcell(sim, 30)
        names = {name for name, _ in CutleryRetrieve(sim, "fork").phases}
        assert MAT_PHASES <= names
        assert ALIGNED_PHASES <= names
    finally:
        sim.close()


def test_fork_retrieval_succeeds_on_a_measured_seed():
    """Seed 30 of the measured 8/10 batch, outputs/fork-retrieve-v8-30-39.json."""
    sim, result = run(30)
    try:
        assert result["status"] == "succeeded", result
        assert result["sustained_lift_s"] >= 0.4
        assert result["stable_release_s"] >= 0.3
        assert result["placement_error_m"] < 0.03
        assert result["drawer_travel_m"] >= 0.085
    finally:
        sim.close()


def test_open_gripper_control_cannot_reach_the_grasp_gate():
    """A gate that cannot fail is not a gate."""
    sim, result = run(30, close_gripper=False)
    try:
        assert result["status"] == "failed"
        assert result["sustained_lift_s"] == 0.0
    finally:
        sim.close()


def test_mats_are_on_opposite_sides_so_the_spoon_needs_both_arms():
    """Measured reach: the left arm cannot place at spoon_mat and the right arm cannot
    reach into the drawer, which is what makes the handoff a physical requirement."""
    assert MATS["fork"][0] < 0 < MATS["spoon"][0]
    assert np.isclose(MATS["fork"][2], MATS["spoon"][2])
