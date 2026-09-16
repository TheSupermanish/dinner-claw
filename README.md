# Bimanual Tabletop VLA

## Current status — 2026-09-16

Five physical skills, each gated on independent physical postconditions and each with
a negative control that must score zero. Full regression in
`outputs/regression-2026-09-16/`:

| skill | arms | seeds | result | open-gripper control |
|---|---|---|---|---|
| Drawer open | left | 30-39 held-out | **10/10** | 0/3 |
| Drawer then fork, one episode, no reset | left | 40-49 held-out | **10/10** | 0/3 |
| **Plate lift, two arms** | both | 50-59 held-out | **9/10** | 0/4 |
| Plate lift, two arms | both | 40-49 tuning | 9/10 | |
| Cup place, scripted teacher | right | 200-209 | **10/10** | 0/3 |
| Cup place, camera-guided | right | 500-549 | **50/50** | |
| Plate lift, ONE arm | left | 40-49 | **0/10**, kept deliberately | |

**The two-arm plate lift is the cooperative action.** Both arms pinch opposing rim
segments and carry the plate level. A single arm grips 94 mm from the plate's centre
of mass, so it applies almost pure torque: it levers the plate onto its far rim and on
six of ten seeds never gets it off the table at all. That single-arm teacher is kept in
the tree at 0/10 as the evidence for why the cell needs two arms, not deleted to tidy
the results.

Say 9/10 for the two-arm lift, not 10/10. Across every configuration measured it lifts
and carries level on 20 of 20 seeds and fails the RELEASE on 2, and which seed fails
moves when unrelated parts of the scene change. 10/10 has been observed; it is not the
behaviour.

Success is never "the script finished". Each skill requires force-bearing contact from
the named gripper pads, a sustained lift with nothing outside the robot bearing the
weight, and a stable released placement at resting height. The two-arm lift adds a
15-degree levelness bound, which a levering pivot cannot satisfy however high it raises
the plate's centre. Failed results are preserved rather than removed.

**Not done:** the spoon handoff (gates are wired and require a measured receiver lift,
but no passing run exists), pouring (the bottle now has a graspable 32 mm neck, but
there is no pour skill and no liquid in this scene), chaining beyond drawer to fork,
dependable VLM reasoning, INT8 quantization, and the qualifying Intel Core Ultra
Series 2/3 run.

**No Intel hardware result exists.** Everything here ran on Apple silicon. Mac CPU
inference numbers are diagnostics and are not challenge results. SmolVLM's local scene
descriptions were unreliable and it issues no motor commands, so the planner is a
deterministic grammar that fails closed, not a VLM. ACT FP32 export passed numerical
fixtures; the FP16 candidate failed parity and is rejected.

### Run it

```bash
uv sync --extra dev --extra train --extra intel --extra reasoning
uv run mjpython -m tabletop_vla.native --seed 0   # native viewer, digit hotkeys
uv run --no-sync tabletop-web --port 8771         # browser console
```

See [`DEMO.md`](DEMO.md) for the demo runbook, the hotkeys, and the claims that are
safe to make. Hotkeys are digits: MuJoCo binds every letter A-Z to its own render
toggles.

Hackathon-scoped implementation for the Intel Physical AI online challenge:
two simulated SO-101 arms in MuJoCo, an observable language-to-skill planner,
closed-loop skill execution, ten-seed evaluation, and OpenVINO benchmarking.

## Scope that can finish

The first demo is deliberately narrower than the full dinner-table prompt:

1. Open the drawer.
2. Pick and place the plate and mug.
3. Perform one fork handoff between arms.
4. Re-check scene state after every skill and stop or replan on failure.
5. Repeat the same scenario over ten randomized seeds.

The low-level baseline should be deterministic IK and guarded waypoints. ACT or
SmolVLA follows behind the same skill interface once a reliable teacher exists.
The brief asks for policy training/fine-tuning; classical control alone does not
complete that objective. Detailed acceptance criteria and dependencies are in `tasks.md`.

## Open the live local console

```bash
uv sync --extra dev
uv run tabletop-web --port 8771
```

Open http://127.0.0.1:8771 on this Mac. **Pick & place cup** executes an approximately
18-second physical skill. Change the seed to vary the starting scene. Play/pause physics, step, switch overview/top
camera, select an arm, move joint targets, reset a seed, or test a small upward reach.
This is a browser interface to local MuJoCo, not a hosted service. Keep the command
running while using it. The console does not yet execute the dinner-table task.

Camera frames come directly from MuJoCo. Seeded reset changes object placement, mass,
friction, and lighting. Sliders command actual limited motors. The reach tool solves
position-only IK in scratch data and then commands motors; it does not teleport arms.
The cup skill checks force-bearing contact with both fingers, a continuous lift of
at least 25 mm for 0.4 seconds, and stable upright placement after release. The
teacher does not weld objects, teleport them, or increase official motor force limits.
It reads simulator state and is **not** the camera policy or the full challenge.

Shape variation currently means bounded rectangular-cup aspect ratios; background
variation means table/floor colors. Arbitrary cup topology and background clutter
are untested. The 50 mm hollow cup and convex jaw collision proxies are documented
simplifications; robot visuals/inertias/motor limits come from the pinned model.
Elliptic contacts with `impratio=10` and three no-slip iterations reduce the
soft-contact slip observed in early tests. See MuJoCo's [grasping guidance](https://mujoco.readthedocs.io/en/latest/modeling.html).

## Native 3D simulator

```bash
uv run mjpython -m tabletop_vla.native
```

On macOS `mjpython` is required for the native viewer. `G` starts cup pickup,
`P` pauses, `D` runs the motor-only demo, `M` switches to manual control, and `R`
resets. Mouse controls rotate/zoom. Native and browser viewers are separate
simulation instances, so their controls are not synchronized.

## Physical evaluation and demonstration recording

```bash
uv run tabletop-eval --seeds 200:210 --output outputs/cup-evaluation.json
uv run tabletop-eval --seeds 0:10 --open-gripper --output outputs/negative-control.json
uv run tabletop-record --seeds 1000:1010 --split train \
  --output outputs/my-training-data --video outputs/ten-seed-cup.mp4
```

Ranges exclude their upper endpoint. Recording requires `ffmpeg` only for video.
Each episode stores synchronized RGB, 12 joint positions, 12 actuator targets,
timestamps, phase labels and outcome; rejected/failed episodes remain labelled.
The recorder refuses to overwrite an existing output directory. This video
demonstrates the cup skill only, not the full dinner-table challenge.

## Official ACT training and independent rollouts

```bash
uv sync --extra dev --extra train
uv run --extra train tabletop-record --seeds 1100:1103 --split validation \
  --output outputs/my-validation-data
uv run --extra train tabletop-train-act --train outputs/my-training-data \
  --validation outputs/my-validation-data --output outputs/my-act --steps 1000
uv run --extra train tabletop-eval-act outputs/my-act --seeds 2000:2010 \
  --output outputs/my-act-evaluation.json
```

Training uses **LeRobot 0.6.0 ACT**, a frozen pretrained ResNet18, RGB plus joint
positions, and 20-action chunks. Normalization uses training episodes only.
Validation/test seeds cannot overlap training. This is an experimental first
baseline; a saved checkpoint or low offline loss does not imply physical success.
The policy evaluator gives the policy no object poses, teacher phase, or IK targets;
ground-truth state is isolated to the scorer. The live app keeps the proven teacher
until learned-policy rollout results justify promotion.

## Architecture

```text
instruction + overview camera
           |
  constrained task planner
           |
 validated skill sequence
           |
 scene-state verifier <----+
           |               |
 dual-arm IK/waypoints ----+
           |
          MuJoCo

VLM/perception model -> OpenVINO IR -> CPU/iGPU/NPU benchmark
```

## Current scaffold

- Official SO-101 MuJoCo model and meshes are pinned under `vendor/so101/`.
- `tabletop-build-scene` generates a namespaced dual-arm scene.
- `tabletop-smoke` loads the scene and advances physics.
- `tabletop-plan` emits a limited keyword-based skill preview, not general language understanding.
- `tabletop-openvino-bench` requires a real recorded input fixture and reports model/device metadata.
- Tests cover physical pickup, negative controls, limits, reset repeatability, HTTP playback and planning.

## Setup

```bash
uv sync --extra dev
uv run tabletop-build-scene
uv run tabletop-smoke
uv run tabletop-render
uv run tabletop-plan "Open the drawer, place the plate and mug, then hand the fork to the left arm"
uv run pytest -q
```

Install the larger Intel toolchain only when a model is ready to convert:

```bash
uv sync --extra intel --extra dev
uv run tabletop-openvino-bench models/openvino/model.xml --inputs models/openvino/example-inputs.npz --device CPU
```

## Next build slice

Promote the trained policy only after it passes held-out physical rollouts; then
extend to drawer/cutlery/plate skills, collision-aware two-arm coordination and
handoff, camera/language reasoning and replanning. Model export/parity and the final
simulation+inference demonstration must be tested on Intel Core Ultra Series 2/3.

## Official resources

- [Intel hackathon resources](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html)
- [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio): ACT/VLA training and OpenVINO export; inspected commit `0388c19b220fe4597da709f26010ea5642586dfe`.
- [OpenVINO Physical AI runtime](https://github.com/openvinotoolkit/physicalai): policy inference and robot/camera interfaces; inspected commit `8ae29a7477c9d31e97382a00971afcb639271373`.
- [LeRobot](https://github.com/huggingface/lerobot): the ACT implementation used locally.

Studio/runtime are official reusable components, not a turnkey dual-arm dinner-table
boilerplate. Their hardware drivers are not required for this Mac simulation.

See `THIRD_PARTY.md` for provenance. Do not claim Intel hardware results until
the benchmark JSON is generated on the required Core Ultra Series 2/3 machine.

## Engineering documentation

- [`docs/ARCHITECTURE_AND_DECISIONS.md`](docs/ARCHITECTURE_AND_DECISIONS.md): exact
  current AI stack, LLM/VLM decision, evidence boundaries, verified competitor-code
  results, and the next implementation slice.
- [`docs/PUBLIC_CODE_AUDIT.md`](docs/PUBLIC_CODE_AUDIT.md): public repositories,
  licenses, reusable ideas, and limitations.
- [`tasks.md`](tasks.md): sequenced requirements checklist and acceptance criteria.
- [`AGENTS.md`](AGENTS.md): truth boundaries and instructions for Codex, Claude,
  OpenCode, or another coding orchestrator.
