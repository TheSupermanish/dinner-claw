from unittest.mock import patch

import numpy as np
import pytest

from tabletop_vla.server import Engine
from tabletop_vla.sim.runtime import Simulation


def test_demo_physically_moves_both_arms_and_completes():
    sim = Simulation()
    try:
        initial = sim.data.qpos.copy()
        sim.start_demo(0)
        np.testing.assert_array_equal(sim.data.qpos[:12], initial[:12])
        for _ in range(9):
            state = sim.step(500)
        assert state["demo"]["status"] == "completed"
        assert state["running"] is False
        assert all(v > 0.08 for v in state["demo"]["peak_tip_displacement_m"].values())
        assert sim.data.warning.number.sum() == 0
        assert all(sim.data.body(obj).xpos[2] > 0.7 for obj in ("plate", "mug", "fork", "spoon"))
    finally:
        sim.close()


def test_server_clock_runs_without_browser_requests_and_pause_stops_it():
    engine = Engine()
    try:
        clock = engine.last_tick
        engine.execute("playback", {"running": True})
        with patch("tabletop_vla.server.time.perf_counter", return_value=clock + 0.05):
            engine.tick()
        assert engine.sim.data.time > 0.04
        engine.execute("playback", {"running": False})
        paused_time = engine.sim.data.time
        with patch("tabletop_vla.server.time.perf_counter", return_value=clock + 1):
            engine.tick()
        assert engine.sim.data.time == paused_time
        engine.execute("control", {"name": "left_shoulder_pan", "value": 0.1})
        assert engine.sim.running
        with pytest.raises(ValueError, match="Pause"):
            engine.execute("step", {"steps": 50})
    finally:
        engine.sim.close()


def test_manual_control_cancels_demo_and_invalid_control_does_not():
    sim = Simulation()
    try:
        sim.start_demo(0)
        with pytest.raises(ValueError):
            sim.control("left_shoulder_pan", 100)
        assert sim.demo["status"] == "running"
        sim.control("left_shoulder_pan", 0.15)
        assert sim.demo["status"] == "cancelled"
        sim.step(100)
        assert sim.data.ctrl[0] == 0.15
    finally:
        sim.close()
