# Repository instructions for coding agents

Read this file, `README.md`, and the top of `tasks.md` before changing code.
This repository is the active Intel Physical AI hackathon project. Work directly in
`/Users/beyond/Desktop/projects/bimanual-tabletop-vla`.

## Objective and truth boundary

Build a reproducible dual-SO-101 MuJoCo dinner-table demo with language/camera task
planning, physical manipulation, randomized-seed evaluation, an ACT/VLA policy, and
OpenVINO deployment. Never present a preview, pose change, low training loss, or
proximity to a target as task success. Success must come from independent physical
postconditions.

Currently verified:

- Right-arm scripted cup pick/place: 50/50 on seeds 200-249.
- Camera-grounded cup teacher with one retry: 20/20 on seeds 600-619.
- Left-arm drawer grip/pull/release: 10/10 on seeds 30-39; open-gripper control 0/3.
- ACT/OpenVINO cup policy exists. Pure OpenVINO result is 7/10 on seeds 2010-2019.
- ACT/OpenVINO plus an explicitly labelled scripted finish is 19/20 on seeds 2020-2039.
- Blank-image hybrid control is 1/10 versus 10/10 normal RGB on matched seeds 2020-2029.
- FP32 export passed fixture parity. FP16-weight export failed action parity and is rejected.
- Mac measurements are diagnostic only. There is no qualifying Intel Core Ultra result.

Not implemented: fork/spoon retrieval, plate placement, object handoff, pouring,
collision-aware full sequencing, reliable VLM planning, and full-task ten-seed evidence.
The cup and drawer currently use separate reset workcells. Planner preview output is
not executable evidence.

## Immediate implementation order

1. Build physical fork and spoon retrieval teachers from an already-open drawer.
2. Build physical plate placement.
3. Build giver-contact -> receiver-contact -> giver-release -> receiver-lift handoff.
4. Add ownership tracking, a shared-workspace lock, collision-checked waypoints, and
   chain drawer -> cutlery -> plate -> cup without resetting between skills.
5. Record successful RGB/proprio/action demonstrations with disjoint seed splits.
6. Train only after teachers pass failure controls and at least ten randomized seeds.
7. Add a strict schema for language/camera reasoning; do not allow an unreliable VLM
   to command motors.
8. Export the selected deployed policy to OpenVINO, verify numerical and closed-loop
   parity, then repeat simulation plus inference on Core Ultra Series 2/3.

Do not spend more time retraining the cup model unless a measured regression justifies
it. The efficient hackathon architecture is hierarchical: a validated task planner
selects skills; reliable scripted or learned controllers execute them; independent
scorers verify postconditions. It is acceptable for some skills to remain scripted as
long as trained-policy and non-trained results are labelled honestly.

## Safety and simulation integrity

- Preserve upstream SO-101 actuator ranges, gains, force limits, inertias, and visuals.
- Never weld or teleport a grasped object, directly drive the drawer joint, or move
  objects during task execution. Workcell setup may reposition objects only before an
  episode and must be declared in its result.
- Keep simulator ground truth out of learned-policy observations. It may be used by a
  teacher or an independent scorer, with that boundary documented.
- Require force-bearing bilateral finger contact, sustained lift/pull, stable release,
  and negative controls. Preserve failed reports instead of overwriting them.
- Reject unsupported language commands completely; never execute a supported prefix
  while silently dropping an unsupported clause.
- Do not claim Intel validation from Apple hardware.

## Commands

Use the existing uv environment and avoid dependency churn:

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
node --check src/tabletop_vla/console.js
uv run --no-sync tabletop-web --port 8771
uv run --no-sync mjpython -m tabletop_vla.native
```

Native controls: `N` drawer, `G` scripted cup, `V` camera cup, `A` pure ACT/OpenVINO,
`H` ACT plus scripted finish, `D` arm-motion diagnostic, `P` pause, `M` manual, `R` reset.
Browser and native viewers are separate MuJoCo instances.

Training flow:

```bash
uv run --no-sync tabletop-record --task TASK --seeds START:STOP --split train --output NEW_DIR
uv run --no-sync tabletop-record --task TASK --seeds START:STOP --split validation --output NEW_DIR
uv run --no-sync tabletop-train-act --train TRAIN_DIR --validation VALIDATION_DIR \
  --output NEW_MODEL_DIR --steps 2500 --chunk-size 50 --action-steps 25
uv run --no-sync tabletop-eval-act NEW_MODEL_DIR --seeds START:STOP --output NEW_REPORT.json
```

At present `tabletop-record` supports cup/learned-cup tasks; extend and test it before
using it for drawer or future tasks. Every output directory/path must be new so prior
evidence remains intact. Training and evaluation seeds must not overlap.

## Key files

- `src/tabletop_vla/sim/runtime.py`: task lifecycle and public capabilities.
- `src/tabletop_vla/sim/manipulation.py`: physical cup teacher.
- `src/tabletop_vla/sim/drawer.py`: physical drawer teacher and workcell.
- `src/tabletop_vla/sim/kinematics.py`: scratch-state IK; no live teleportation.
- `src/tabletop_vla/sim/scoring.py`: independent cup scorer.
- `src/tabletop_vla/record.py`, `train_act.py`, `evaluate_act.py`: learning pipeline.
- `src/tabletop_vla/openvino/`: export and benchmark tools.
- `src/tabletop_vla/planning/task_planner.py`: preview plus strict executable gate.
- `tasks.md`: requirement audit, remaining work, and acceptance criteria.

## Repository hygiene

The repository currently has no initial commit and all files may appear untracked.
Treat them as user-owned work. Do not delete or overwrite existing datasets, models,
reports, videos, vendored assets, or changes. Do not commit, push, publish, deploy, or
submit without explicit user authorization. Keep claims in docs synchronized with
actual tests and saved reports.
