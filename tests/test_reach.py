import numpy as np
import pytest

from tabletop_vla.sim.kinematics import solve_down, solve_pose
from tabletop_vla.sim.reach import NAMED_TARGETS, approach_axes, best_axis
from tabletop_vla.sim.runtime import Simulation


def test_approach_axes_are_unit_vectors_and_include_straight_down():
    axes = approach_axes()
    assert "down" in axes
    np.testing.assert_allclose(axes["down"], [0, 0, -1])
    for name, axis in axes.items():
        assert np.linalg.norm(axis) == pytest.approx(1.0), name


def test_mug_target_still_accepted_by_the_right_arm():
    sim = Simulation()
    try:
        sim.reset(200)
        solution = solve_down(sim.model, sim.data, "right", NAMED_TARGETS["mug_mat"])
        assert solution.accepted
        assert solution.position_error < 1e-4
    finally:
        sim.close()


def test_reach_measurement_is_reproducible():
    """A solver whose answer moves between runs cannot serve as evidence."""
    sim = Simulation()
    try:
        sim.reset(200)
        axes = approach_axes(tilts=(0, 30))
        first = best_axis(sim.model, sim.data, "right", NAMED_TARGETS["mug_mat"], axes)
        second = best_axis(sim.model, sim.data, "right", NAMED_TARGETS["mug_mat"], axes)
        assert first == second
    finally:
        sim.close()


def test_solve_pose_rejects_a_degenerate_approach_axis():
    sim = Simulation()
    try:
        sim.reset(200)
        with pytest.raises(ValueError):
            solve_pose(sim.model, sim.data, "right", NAMED_TARGETS["mug_mat"], approach=(0, 0, 0))
    finally:
        sim.close()
