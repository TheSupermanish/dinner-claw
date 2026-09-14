"""Native interactive 3D MuJoCo viewer. On macOS run with mjpython."""
from __future__ import annotations

import argparse
import queue
import time

import glfw
import mujoco
import mujoco.viewer

from tabletop_vla.sim.runtime import Simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # Resolve the display before constructing the native viewer. On this Mac,
    # immediate startup otherwise crashed inside GLFW's Cocoa video-mode lookup;
    # explicitly querying the primary video mode allowed repeatable startup.
    monitor = glfw.get_primary_monitor()
    if not monitor or glfw.get_video_mode(monitor) is None:
        raise RuntimeError("No usable desktop display. Use the live browser console instead.")
    sim = Simulation()
    keys = queue.Queue()
    sim.start_task(args.seed)
    print("Native MuJoCo: N = drawer, F = drawer+fork, L = plate, G = cup teacher, "
          "V = camera, A = ACT, H = ACT+finish, D = demo, P = pause, M = manual, R = reset",
          flush=True)
    error = None
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys.put) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.87]
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 130
                viewer.cam.elevation = -32
            while viewer.is_running():
                start = time.perf_counter()
                with viewer.lock():
                    while not keys.empty():
                        key = keys.get_nowait()
                        if key in (ord("G"), ord("g")):
                            sim.start_task(args.seed)
                            error = None
                        elif chr(key).upper() in {"V", "A", "H", "N", "F", "L"}:
                            names = {"V": "camera-cup-place", "A": "learned-cup-place",
                                     "H": "learned-cup-safe-place", "N": "drawer-open",
                                     # F chains drawer -> fork retrieval in ONE episode
                                     # with no reset: 10/10 on held-out seeds 40-49.
                                     "F": "fork-retrieve",
                                     # L grasps and lifts the plate by its rim on every
                                     # seed; the carry to the mat is not solved yet.
                                     "L": "plate-place-left"}
                            try:
                                sim.start_task(args.seed, names[chr(key).upper()])
                                error = None
                            except (ValueError, RuntimeError, OSError) as exc:
                                error = str(exc)
                        elif key in (ord("D"), ord("d")):
                            sim.start_demo(args.seed)
                        elif key in (ord("P"), ord("p")):
                            sim.running = not sim.running
                        elif key in (ord("M"), ord("m")):
                            sim.cancel_demo()
                            sim.running = True
                        elif key in (ord("R"), ord("r")):
                            sim.reset(args.seed)
                            sim.running = True
                    if sim.running:
                        sim.step(10)
                    demo = sim.demo
                    status = demo["phase"] if demo else "Manual control"
                    detail = "Motor control, not an AI policy"
                    if sim.task:
                        result = sim.task.state()
                        status = result["status"] + ": " + result["phase"]
                        detail = result["reason"] or (f"Drawer opened {result['drawer_travel_m'] * 100:.1f} cm"
                            if result["name"] == "drawer-open" else (
                            f"Lift {result['peak_lift_m'] * 100:.1f} cm | "
                            f"Place error {result['placement_error_m'] * 1000:.1f} mm"
                        ))
                    if not sim.running:
                        status += " (paused)"
                viewer.set_texts((
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    ("SO-101 LIVE PHYSICS\nN: drawer | F: drawer+fork | L: plate\n"
                     "G: cup teacher | V: camera | A: ACT | H: ACT+finish\n"
                     "P: pause | D: demo | M: manual | R: reset"),
                    f"Seed {args.seed} | {sim.data.time:.2f}s\n{status}\n{error or detail}\n" + (
                        sim.task.state()["controller"] if sim.task else "Motor control only"),
                ))
                viewer.sync()
                remaining = 0.02 - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
