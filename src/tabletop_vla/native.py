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
    parser.add_argument("--free-visuals", action="store_true",
                        help="Allow MuJoCo's own letter-key render toggles to stick. "
                             "Off by default so a stray keypress cannot leave contact "
                             "arrows on screen during a demo.")
    args = parser.parse_args()
    # Resolve the display before constructing the native viewer. On this Mac,
    # immediate startup otherwise crashed inside GLFW's Cocoa video-mode lookup;
    # explicitly querying the primary video mode allowed repeatable startup.
    monitor = glfw.get_primary_monitor()
    if not monitor or glfw.get_video_mode(monitor) is None:
        raise RuntimeError("No usable desktop display. Use the live browser console instead.")
    sim = Simulation()
    keys = queue.Queue()
    # Hotkeys are DIGITS, never letters. MuJoCo's viewer binds every letter A-Z to a
    # visualization or rendering toggle of its own (mjVISSTRING and mjRNDSTRING), and
    # `launch_passive` runs that built-in handler ALONGSIDE this callback, so a letter
    # hotkey fires both. Measured 2026-09-16, the old letter map collided on twelve
    # keys: F also toggled Contact Force, which buries the scene in giant arrows; L
    # also toggled Additive rendering, which washes the table out in white; B toggled
    # Perturb Force, N toggled Island, V Tendon, A Auto Connect, H Convex Hull, D
    # Static Body, P Contact Split, M Center of Mass, G Fog and R Reflection. Neither
    # mjVISSTRING nor mjRNDSTRING binds a digit, so digits are collision-free.
    TASK_KEYS = {
        "1": "drawer-open",
        # 2 chains drawer -> fork retrieval in ONE episode with no reset:
        # 10/10 on held-out seeds 40-49.
        "2": "fork-retrieve",
        # 3 lifts the plate with BOTH arms: 10/10 held-out seeds 50-59, control 0/4.
        "3": "bimanual-plate-lift",
        # 4 is the SINGLE-arm plate and it FAILS, 0/10, deliberately. A one-arm rim
        # pinch grips 94 mm off the centre of mass, so it levers the plate up on its
        # far rim; six of ten seeds never leave the table. Run straight after 3 on the
        # same seed it is the clearest evidence for why the cell has two arms.
        "4": "plate-place-left",
        "5": "camera-cup-place",
        "6": "cup-place",
        "7": "learned-cup-place",
        "8": "learned-cup-safe-place",
    }
    # The seed is live, not fixed at startup. Every skill was measured on its OWN
    # range (drawer 30-39, fork 40-49, two-arm plate 50-59, cup 500-549), so a viewer
    # locked to one startup seed either demonstrates seeds nobody evaluated or needs a
    # restart between skills. `-` and `=` step it; the overlay always shows which.
    seed = args.seed
    sim.start_task(seed)
    print("Native MuJoCo (digit keys; letters are MuJoCo's own render toggles): "
          "1 drawer | 2 drawer+fork | 3 two-arm plate | 4 one-arm plate (fails 0/10) | "
          "5 camera cup | 6 cup teacher | 7 ACT | 8 ACT+finish | 9 motor demo | "
          "0 reset | - / = seed | . pause", flush=True)
    error = None
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys.put) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.87]
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 130
                viewer.cam.elevation = -32
            # The clean look, captured before anything can toggle it. Restored every
            # frame unless --free-visuals, so a stray keypress cannot leave contact
            # arrows or perturbation vectors on screen in front of an audience.
            pinned = tuple(viewer.opt.flags)
            # Contact-force and perturbation arrows live in `viewer.opt.flags`, which is
            # public. Additive, Fog, Reflection, Shadow, Wireframe and Skybox live in the
            # scene's own render flags, which the handle does not expose; reach for them
            # defensively, because failing to pin them is cosmetic while raising inside
            # the viewer loop would close the window mid-demo.
            scene = None
            pinned_render = ()
            try:
                candidate = viewer._get_sim()
                if candidate is not None and hasattr(candidate, "scn"):
                    scene = candidate.scn
                    pinned_render = tuple(scene.flags)
            except Exception:  # noqa: BLE001 - never let a private API kill the viewer
                scene = None
            while viewer.is_running():
                start = time.perf_counter()
                with viewer.lock():
                    if not args.free_visuals and tuple(viewer.opt.flags) != pinned:
                        for index, value in enumerate(pinned):
                            viewer.opt.flags[index] = value
                    if (not args.free_visuals and scene is not None
                            and tuple(scene.flags) != pinned_render):
                        try:
                            for index, value in enumerate(pinned_render):
                                scene.flags[index] = value
                        except Exception:  # noqa: BLE001
                            scene = None
                    while not keys.empty():
                        key = keys.get_nowait()
                        name = TASK_KEYS.get(chr(key) if 0 <= key < 0x110000 else "")
                        if name is not None:
                            try:
                                sim.start_task(seed, name)
                                error = None
                            except (ValueError, RuntimeError, OSError) as exc:
                                error = str(exc)
                        elif key == ord("9"):
                            sim.start_demo(seed)
                        elif key == ord("."):
                            sim.running = not sim.running
                        elif key == ord("0"):
                            sim.cancel_demo()
                            sim.reset(seed)
                            sim.running = True
                            error = None
                        elif key in (ord("-"), ord("=")):
                            seed = max(0, seed + (1 if key == ord("=") else -1))
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
                     ("SO-101 LIVE PHYSICS   (digit keys)\n"
                     "1 drawer 30-39 | 2 drawer+fork 40-49 | 3 two-arm plate 50-59\n"
                     "4 one-arm plate (fails 0/10) | 5/6 cup | 7/8 ACT | 9 motor demo\n"
                     ". pause | 0 reset | - / = seed"),
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
