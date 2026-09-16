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

### Hotkeys are DIGITS. Do not press letters.

MuJoCo's viewer binds every letter A-Z to one of its own render toggles, and it runs
that handler alongside ours, so a letter does two things at once. `F` turns on Contact
Force and fills the scene with giant arrows; `L` turns on Additive and washes the table
out in white; `B`, `N`, `V`, `A`, `H`, `D`, `P`, `M`, `G`, `R` all toggle something.
Our hotkeys were letters until 2026-09-16 and collided on twelve of them.

Digits are bound by neither `mjVISSTRING` nor `mjRNDSTRING`, so they are collision-free
and that is what the viewer uses now. The visual flags are also pinned every frame, so
even a stray letter cannot leave arrows on screen. Pass `--free-visuals` if you
deliberately want MuJoCo's own toggles (wireframe, transparency) to stick.

If arrows or a white wash ever do appear, the key that turned it on turns it off:
press `F` again for arrows, `L` for the white wash, `G` for fog, `R` for reflections.

Camera starts at azimuth 130, elevation -32, distance 1.6, looking at the table
centre. Drag to orbit. The two-arm plate lift reads best from a low angle where the
plate's tilt is visible against the table edge.

## Seeds

Each skill was evaluated on its own range, and the viewer shows the live seed in the
top-right overlay. `-` and `=` step it without restarting.

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

### 1. `1` — drawer, 18 s
The left arm grips the handle, pulls, releases, retracts. Overlay shows
`Drawer opened 9.9 cm`.

Worth saying: success is not "the script finished". The drawer must travel at least
8.5 cm **through a verified bilateral handle pull**, then stay open with the whole arm
off the whole drawer. An arm still leaning on the drawer front fails the gate.

### 2. `2` — drawer then fork, 46 s
The longest one. The drawer stage runs again, and then the fork is retrieved **in the
same episode with no reset**, no weld, no teleport. That is the point of this skill:
the second half depends on the physical state the first half produced.

It is 46 seconds of watching, so fill it with the no-reset explanation.

### 3. `3` — two-arm plate lift, 19 s
The headline. Both arms grip opposing rim segments and carry the plate to its mat.

Overlay reads:
```
Lift 8.1 cm | Tilt 13.5 deg of 15 | 4-pad 3.20 s | Place error 20.7 mm
```
Point at **Tilt**. That is what makes this a lift rather than a lever, and it is a
success condition: over 15 degrees at any airborne moment and the episode fails.

### 4. `4` — one-arm plate, 20 s, and it fails
Run it straight after `3`, same seed. The single arm pinches the rim 94 mm from the
plate's centre of mass, so it applies almost pure torque: the plate tips up on its far
rim and never leaves the table. Six of ten seeds do not lift at all. **0/10.**

This is the most honest 20 seconds in the demo and the clearest argument for why the
cell has two arms. It is kept in the tree deliberately, not hidden.

### 5. `5` — camera-guided cup, 18 s
Locates the cup from top-camera pixels, then runs the scripted skill. Calibrated
colour vision and a strict command grammar, **not a VLM**.

### The green cylinder has no demo
It is the bottle. There is no pour skill, and MuJoCo has no liquid here. It is set
dressing plus a planner fixture: the rule-based planner will PREVIEW a pour as
`pick(left, bottle) -> pour(left, bottle)`, and `executable_task` then refuses it,
because `EXECUTABLE` holds only the three skills with measured held-out rates
(`camera-cup-place`, `fork-retrieve`, `bimanual-plate-lift`). If someone asks the
system to pour, it fails closed rather than flailing. That refusal is worth showing
if the planner comes up, but it lives in the CLI and web console, not this viewer.

## Full key map

```
1 drawer          2 drawer+fork      3 two-arm plate    4 one-arm plate (fails)
5 camera cup      6 cup teacher      7 ACT              8 ACT+finish
9 motor demo      0 reset            . pause            - / =  seed down / up
```

Use the number row, not the numeric keypad: the keypad sends different key codes.

## Physics keeps running after an episode ends

Every teacher stops itself on success or failure. The viewer used to stop stepping
with it, so the final frame froze: a failed one-arm plate lift stayed propped at 25
degrees and looked like gravity had been switched off. Physics now continues and only
`.` pauses it.

That matters for honesty as much as for looks. When the script stops steering, anything
it was merely holding up falls, on camera. Measured on seed 50 after the one-arm plate
fails: the plate drops from z 0.780 to 0.770 and relaxes from 24.6 to 18.2 degrees over
three seconds, and the recorded result stays `failed`. The overlay says
`(episode over, physics still running)` so nobody mistakes the settling for the task.

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
2. `0` resets the same seed; `-` / `=` move to another; `.` pauses.
3. A failure on an unevaluated seed is not a contradiction of a held-out number, and
   saying so is better than pretending it did not happen.

The single-arm plate (`4`) fails every time by design. Do not "fix" it live.
