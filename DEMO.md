# Demo runbook: native MuJoCo viewer

Everything below was measured on this machine (Apple silicon, MuJoCo 3.13.0). Numbers
are quoted from `outputs/*.json`. Nothing here is an Intel result.

## Launch

```bash
uv run mjpython -m tabletop_vla.native --seed 0
```

`mjpython` is required on macOS; plain `python` will not open a viewer window. The
viewer runs at **1.0x realtime**, so an episode takes as long on the clock as its
simulated duration.

Camera starts at azimuth 130, elevation -32, distance 1.6, looking at the table
centre. Drag to orbit. The two-arm plate lift reads best from a low angle where the
plate's tilt is visible against the table edge.

## Seeds

Each skill was evaluated on its own range, and the viewer shows the live seed in the
top-right overlay. `[` and `]` step it without restarting.

| skill | evaluated range | result |
|---|---|---|
| drawer open | 30-39 held-out | 10/10, travel 9.95-9.96 cm, open-gripper control 0/3 |
| drawer + fork | 40-49 held-out | 10/10, placement error 4.6-17.4 mm |
| two-arm plate | 50-59 held-out | 10/10; 9/10 on tuning seeds 40-49; control 0/4 |
| camera cup | 500-549 | 49/50; 20/20 on 600-619; 40/50 on 400-449 |

**Seed 0 passes all four** and is the default. Verified 2026-09-16 by running every
skill end to end at seeds 0-3: seeds 0 and 2 give 4/4, seeds 1 and 3 give 3/4 with the
camera cup failing its grasp. If you improvise a seed live, the cup is the one that
will bite you. The drawer, fork and two-arm plate passed on all four seeds tried.

## Suggested order

Times are wall-clock, at 1.0x.

### 1. `N` — drawer, 18 s
The left arm grips the handle, pulls, releases, retracts. Overlay shows
`Drawer opened 9.9 cm`.

Worth saying: success is not "the script finished". The drawer must travel at least
8.5 cm **through a verified bilateral handle pull**, then stay open with the whole arm
off the whole drawer. An arm still leaning on the drawer front fails the gate.

### 2. `F` — drawer then fork, 46 s
The longest one. The drawer stage runs again, and then the fork is retrieved **in the
same episode with no reset**, no weld, no teleport. That is the point of this skill:
the second half depends on the physical state the first half produced.

It is 46 seconds of watching, so fill it with the no-reset explanation.

### 3. `B` — two-arm plate lift, 19 s
The headline. Both arms grip opposing rim segments and carry the plate to its mat.

Overlay reads:
```
Lift 8.1 cm | Tilt 13.5 deg of 15 | 4-pad 3.20 s | Place error 20.7 mm
```
Point at **Tilt**. That is what makes this a lift rather than a lever, and it is a
success condition: over 15 degrees at any airborne moment and the episode fails.

### 4. `L` — one-arm plate, 20 s, and it fails
Run it straight after `B`, same seed. The single arm pinches the rim 94 mm from the
plate's centre of mass, so it applies almost pure torque: the plate tips up on its far
rim and never leaves the table. Six of ten seeds do not lift at all. **0/10.**

This is the most honest 20 seconds in the demo and the clearest argument for why the
cell has two arms. It is kept in the tree deliberately, not hidden.

### 5. `V` — camera-guided cup, 18 s
Locates the cup from top-camera pixels, then runs the scripted skill. Calibrated
colour vision and a strict command grammar, **not a VLM**.

## What the overlay means

- `status: phase` — live task state and current phase name.
- Second line — the failure reason if it failed, otherwise per-skill telemetry.
- Last line — which controller is driving, so a scripted teacher is never mistaken
  for a learned policy.

## Claims that are safe, and claims that are not

Safe, because they are measured and reproducible from `outputs/`:
- physical contact-gated success, negative controls at 0, held-out seed evaluation
- the drawer-to-fork chain with no reset between skills
- the two-arm plate lift, including its levelness bound
- ACT trained and exported to OpenVINO FP32, running locally

**Not safe, do not say:**
- any Intel Core Ultra latency or throughput figure. Nothing here ran on Intel
  hardware. Mac CPU numbers are diagnostics, not challenge results.
- "the VLM plans the task". SmolVLM's scene descriptions were unreliable and it
  issues no motor commands. The planner is a rule-based preview.
- "full dinner-table sequence". The spoon handoff is not implemented, and the skills
  are not chained beyond drawer to fork.
- FP16: the candidate failed numerical parity and is rejected. FP32 only.

## If something fails live

It can. The gates are strict on purpose and nothing is stage-managed.

1. Say what the overlay says. The failure reason is specific and is the real output.
2. `R` resets the same seed; `[` / `]` move to another.
3. A failure on an unevaluated seed is not a contradiction of a held-out number, and
   saying so is better than pretending it did not happen.

The single-arm plate (`L`) fails every time by design. Do not "fix" it live.
