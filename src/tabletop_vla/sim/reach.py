"""Measured IK reach map. Solver convergence only; collisions are not checked."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tabletop_vla.sim.kinematics import solve_pose

# Fork/spoon world positions are taken after a measured drawer opening, not guessed.
NAMED_TARGETS = {
    "fork_open": [-0.274838, 0.070052, 0.763889],
    "spoon_open": [-0.184839, 0.069895, 0.763889],
    "plate": [-0.20, 0.12, 0.76],
    "fork_mat": [-0.32, 0.10, 0.75],
    "spoon_mat": [0.32, 0.10, 0.75],
    "mug_mat": [0.27, -0.04, 0.75],
    "handoff": [0.00, -0.05, 0.85],
}


def approach_axes(tilts=(0, 15, 30, 45, 60, 75, 90), azimuths=8):
    """Unit finger-pointing axes on cones around straight down."""
    axes = {}
    for tilt in tilts:
        rad = np.deg2rad(tilt)
        if tilt == 0:
            axes["down"] = np.array([0.0, 0.0, -1.0])
            continue
        for k in range(azimuths):
            az = 2 * np.pi * k / azimuths
            axes[f"t{tilt}_a{round(np.rad2deg(az))}"] = np.array(
                [np.sin(rad) * np.cos(az), np.sin(rad) * np.sin(az), -np.cos(rad)]
            )
    return axes


def _spherical(axis):
    """Tilt from straight down, and azimuth, both in degrees."""
    axis = np.asarray(axis, dtype=float)
    tilt = np.rad2deg(np.arccos(np.clip(-axis[2], -1.0, 1.0)))
    azimuth = np.rad2deg(np.arctan2(axis[1], axis[0]))
    return float(tilt), float(azimuth)


def _cartesian(tilt_deg, azimuth_deg):
    tilt, azimuth = np.deg2rad(tilt_deg), np.deg2rad(azimuth_deg)
    return np.array([np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth),
                     -np.cos(tilt)])


def refine_axes(axis, span=12.0, step=2.0):
    """A local axis grid around a coarse winner, so a near miss gets a real answer."""
    tilt, azimuth = _spherical(axis)
    out = {}
    for dt in np.arange(-span, span + 1e-9, step):
        for da in np.arange(-span * 2, span * 2 + 1e-9, step * 2):
            t = float(np.clip(tilt + dt, 0.0, 90.0))
            a = float(azimuth + da)
            out[f"r_t{t:.0f}_a{a:.0f}"] = _cartesian(t, a)
    return out


def best_axis(model, data, arm, target, axes):
    """Best solver result over every approach axis, preferring an accepted one."""
    best = None
    for name, axis in axes.items():
        solution = solve_pose(model, data, arm, target, approach=axis)
        score = (not solution.accepted, solution.position_error + 0.1 * solution.axis_error)
        if best is None or score < best[0]:
            best = (score, name, axis, solution)
    _, name, axis, solution = best
    return {"arm": arm, "approach": name, "axis": [round(v, 4) for v in axis],
            "position_error_m": solution.position_error, "axis_error": solution.axis_error,
            "accepted": bool(solution.accepted)}


def sweep_named(sim, axes, targets=None, refine=True):
    rows = []
    for label, target in (targets or NAMED_TARGETS).items():
        for arm in ("left", "right"):
            row = best_axis(sim.model, sim.data, arm, target, axes)
            if refine and not row["accepted"]:
                local = best_axis(sim.model, sim.data, arm, target, refine_axes(row["axis"]))
                score = (local["position_error_m"] + 0.1 * local["axis_error"],)
                if local["accepted"] or score < (row["position_error_m"] + 0.1 * row["axis_error"],):
                    local["refined_from"] = row["approach"]
                    row = local
            rows.append({"target": label, "target_xyz": list(map(float, target)), **row})
    return rows


def sweep_grid(sim, axes, step, bounds):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    rows = []
    for x in np.arange(x0, x1 + 1e-9, step):
        for y in np.arange(y0, y1 + 1e-9, step):
            for z in np.arange(z0, z1 + 1e-9, step):
                target = [float(x), float(y), float(z)]
                for arm in ("left", "right"):
                    row = best_axis(sim.model, sim.data, arm, target, axes)
                    if row["accepted"]:
                        rows.append({"target_xyz": target, **row})
    return rows


def main():
    from tabletop_vla.sim.runtime import Simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--drawer-workcell", action="store_true",
                        help="Apply the cabinet workcell so the cutlery targets are in scene")
    parser.add_argument("--grid", action="store_true", help="Also sweep a coarse tabletop grid")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--tilts", default="0,15,30,45,60,75,90")
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip the local axis refinement around each coarse winner")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; existing evidence is never overwritten")

    axes = approach_axes(tuple(int(t) for t in args.tilts.split(",")))
    sim = Simulation()
    started = time.perf_counter()
    try:
        sim.reset(args.seed)
        if args.drawer_workcell:
            from tabletop_vla.sim.drawer import prepare_workcell

            prepare_workcell(sim, args.seed)
        named = sweep_named(sim, axes, refine=not args.no_refine)
        grid = sweep_grid(sim, axes, args.grid_step,
                          ((-0.45, 0.45), (-0.15, 0.30), (0.75, 0.90))) if args.grid else []
    finally:
        sim.close()

    report = {"seed": args.seed, "drawer_workcell": args.drawer_workcell,
              "approach_axes": len(axes), "tilts_deg": args.tilts,
              "refined": not args.no_refine,
              "acceptance": "position_error < 0.003 m and axis_error < 0.04",
              "limitation": "IK convergence only; no collision or torque check",
              "elapsed_s": time.perf_counter() - started,
              "named": named, "grid_accepted": grid,
              "grid_step_m": args.grid_step if args.grid else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in named:
        print(f"{row['target']:11s} {row['arm']:5s} {row['approach']:10s} "
              f"pe={row['position_error_m']:.5f} ae={row['axis_error']:.4f} "
              f"accepted={row['accepted']}", flush=True)
    print(f"accepted grid cells: {len(grid)}  report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
