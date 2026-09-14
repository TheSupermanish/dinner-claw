import pytest

from tabletop_vla.sim.drawer import prepare_workcell
from tabletop_vla.sim.handoff import (
    ALIGNED_PHASES,
    LEFT_GRIPPING,
    LEFT_PHASES,
    RIGHT_GRIPPING,
    RIGHT_PHASES,
    SpoonHandoff,
)
from tabletop_vla.sim.runtime import Simulation


def run(seed, close_gripper=True, batches=90):
    sim = Simulation()
    sim.reset(seed)
    prepare_workcell(sim, seed)
    sim.task = SpoonHandoff(sim, close_gripper=close_gripper)
    sim.running = True
    for _ in range(batches):
        sim.step(500)
        if sim.task.status != "running":
            break
    return sim, sim.task.state()


def test_runtime_whitelists_spoon_handoff():
    """runtime.py may only gain the task name plus construction; nothing else."""
    sim = Simulation()
    try:
        sim.start_task(30, "spoon-handoff")
        assert sim.task.state()["name"] == "spoon-handoff"
        assert sim.task.state()["stage"] == "giver"
    finally:
        sim.close()


def test_runtime_still_rejects_an_unimplemented_task_name():
    sim = Simulation()
    try:
        with pytest.raises(ValueError):
            sim.start_task(30, "plate-place")
    finally:
        sim.close()


def test_phase_sets_name_real_phases():
    """These sets were substring matches once in cutlery.py, and 'Release' silently
    fell through to the wrong wrist azimuth. Every name in them must exist."""
    sim = Simulation()
    try:
        sim.reset(30)
        prepare_workcell(sim, 30)
        names = {name for name, _ in SpoonHandoff(sim).phases}
        assert LEFT_PHASES <= names
        assert RIGHT_PHASES <= names
        assert ALIGNED_PHASES <= names
        assert LEFT_GRIPPING <= names
        assert RIGHT_GRIPPING <= names
    finally:
        sim.close()


def test_handoff_fails_honestly_without_transfer_on_seed_30():
    """Stage-1 probe: the 36 mm handle fits one gripper, not two, and the receiver
    cannot reach a grasp-aligned pose (arm-to-arm 18-24 N). The giver must still
    hold and lift, gate 1 passes, and gates 2-5 stay failed in order."""
    sim, result = run(30)
    try:
        assert result["status"] == "failed", result
        assert result["gates"]["giver_holds"] is True
        assert result["giver_bilateral_s"] >= 0.15
        assert result["giver_sustained_lift_s"] >= 0.4
        assert result["gates"]["receiver_contacts"] is False
        assert result["gates"]["giver_releases"] is False
        assert result["gates"]["receiver_lifts"] is False
        assert result["gates"]["sustained_ownership"] is False
        assert result["simultaneous_bilateral_s"] < 0.15
        assert result["receiver_only_s"] == 0.0
    finally:
        sim.close()


def test_open_right_gripper_control_cannot_pass_receiver_gates():
    """A gate that cannot fail is not a gate: right held open must score 0, with
    the giver unaffected so the failure lands on the receiver gates, not gate 1."""
    sim, result = run(30, close_gripper=False)
    try:
        assert result["status"] == "failed"
        assert result["gates"]["giver_holds"] is True
        assert result["gates"]["receiver_contacts"] is False
        assert result["simultaneous_bilateral_s"] == 0.0
        assert result["receiver_only_s"] == 0.0
    finally:
        sim.close()
