import numpy as np
import pytest

from tabletop_vla.sim.runtime import Simulation


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_left_arm_physically_opens_and_releases_drawer(seed):
    sim = Simulation()
    try:
        sim.reset(seed)
        baseline_geoms = sim.model.geom_pos.copy()
        sim.start_task(seed, "drawer-open")
        assert sim.model.neq == 0
        assert sim.model.joint("drawer_slide").id not in sim.model.actuator_trnid[:, 0]
        for _ in range(20):
            sim.step(500)
            if sim.task.status != "running":
                break
        result = sim.task.state()
        assert result["status"] == "succeeded", result
        assert result["drawer_travel_m"] >= 0.085
        assert result["contact_pull_m"] >= 0.06
        assert result["bilateral_contact_s"] >= 0.15
        assert result["stable_released_s"] >= 0.3
        assert not sim.data.warning.number.any()
        sim.reset(seed)
        np.testing.assert_array_equal(sim.model.body_pos, sim.original_body_pos)
        np.testing.assert_array_equal(sim.model.geom_pos, baseline_geoms)
        assert sim.model.opt.disableflags == sim.original_disableflags
    finally:
        sim.close()


def test_open_fingers_cannot_claim_drawer_success():
    sim = Simulation()
    try:
        sim.start_task(20, "drawer-open", close_gripper=False)
        for _ in range(20):
            sim.step(500)
            if sim.task.status != "running":
                break
        assert sim.task.status == "failed"
        assert "grasp" in sim.task.reason
        assert sim.task.contact_pull_m < 0.06
    finally:
        sim.close()
