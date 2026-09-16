"""Native interactive 3D MuJoCo viewer. On macOS run with mjpython."""
from __future__ import annotations

import argparse
import queue
import time

import glfw
import mujoco
import mujoco.viewer

from tabletop_vla.sim.runtime import Simulation


def _telemetry(result):
    """The line under the status, chosen to show what each skill is judged on.

    The two-arm plate is judged on LEVELNESS, so a viewer that shows only lift
    height and placement error hides the number that separates lifting the plate
    from levering it up on its far rim. Tilt goes on screen for that task.
    """
    if result["name"] == "drawer-open":
        return f"Drawer opened {result['drawer_travel_m'] * 100:.1f} cm"
    lift = f"Lift {result['peak_lift_m'] * 100:.1f} cm"
    place = f"Place error {result['placement_error_m'] * 1000:.1f} mm"
    if result["name"] == "bimanual-plate-lift":
        return (f"{lift} | Tilt {result['peak_tilt_during_lift_deg']:.1f} deg of 15 "
                f"| 4-pad {result['four_pad_contact_s']:.2f} s | {place}")
    return f"{lift} | {place}"


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
    # The seed is live, not fixed at startup. Every skill was measured on its OWN
    # range (drawer 30-39, fork 40-49, two-arm plate 50-59, cup 200-209), so a viewer
    # locked to one startup seed either demonstrates seeds nobody evaluated or needs a
    # restart between skills. `[` and `]` step it; the overlay always shows which.
    seed = args.seed
    sim.start_task(seed)
    print("Native MuJoCo: N = drawer, F = drawer+fork, L = plate (fails, 0/10), "
          "B = bimanual plate, G = cup teacher, V = camera, A = ACT, H = ACT+finish, "
          "D = demo, P = pause, M = manual, R = reset, [ / ] = seed -/+", flush=True)
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
                            sim.start_task(seed)
                            error = None
                        elif chr(key).upper() in {"V", "A", "H", "N", "F", "L", "B"}:
                            names = {"V": "camera-cup-place", "A": "learned-cup-place",
                                     "H": "learned-cup-safe-place", "N": "drawer-open",
                                     # F chains drawer -> fork retrieval in ONE episode
                                     # with no reset: 10/10 on held-out seeds 40-49.
                                     "F": "fork-retrieve",
                                     # L is the SINGLE-arm plate and it fails: 0/10.
                                     # Kept as the contrast for B. A one-arm rim pinch
                                     # grips 94 mm off the centre of mass, so it levers
                                     # the plate up on its far rim instead of lifting;
                                     # six of ten seeds never leave the table at all.
                                     "L": "plate-place-left",
                                     # B lifts the plate with BOTH arms to its mat:
                                     # 10/10 on held-out seeds 50-59, open-gripper 0/4.
                                     "B": "bimanual-plate-lift"}
                            try:
                                sim.start_task(seed, names[chr(key).upper()])
                                error = None
                            except (ValueError, RuntimeError, OSError) as exc:
                                error = str(exc)
                        elif key in (ord("D"), ord("d")):
                            sim.start_demo(seed)
                        elif key in (ord("P"), ord("p")):
                            sim.running = not sim.running
                        elif key in (ord("M"), ord("m")):
                            sim.cancel_demo()
                            sim.running = True
                        elif key in (ord("R"), ord("r")):
                            sim.reset(seed)
                            sim.running = True
                        elif key in (ord("["), ord("]")):
                            seed = max(0, seed + (1 if key == ord("]") else -1))
                            sim.reset(seed)
                            sim.running = True
                            error = None
                    if sim.running:
                        sim.step(10)
                    demo = sim.demo
                    status = demo["phase"] if demo else "Manual control"
                    detail = "Motor control, not an AI policy"
                    if sim.task:
                        result = sim.task.state()
                        status = result["status"] + ": " + result["phase"]
                        detail = result["reason"] or _telemetry(result)
                    if not sim.running:
                        status += " (paused)"
                viewer.set_texts((
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     ("SO-101 LIVE PHYSICS\n"
                     "N: drawer 30-39 | F: drawer+fork 40-49 | B: two-arm plate 50-59\n"
                     "L: one-arm plate (fails 0/10) | G/V: cup | A/H: ACT\n"
                     "P: pause | D: demo | M: manual | R: reset | [ ]: seed"),
                    f"Seed {seed} | {sim.data.time:.2f}s\n{status}\n{error or detail}\n" + (
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
