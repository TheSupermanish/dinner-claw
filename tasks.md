# Build plan and requirements audit

## Latest implementation checkpoint

- [x] Freeze further cup training to focus usage on missing physical skills.
- [x] Implement left-arm drawer grip/pull/release/retract in a reachable cabinet workcell.
- [x] Require bilateral handle contact, >=6 cm contact-supported pull, >=8.5 cm final
  opening, and >=0.3 s released stability. Add open-gripper failure control.
- [x] Add browser drawer control and native `N` shortcut.
- [x] Keep camera teacher, pure ACT and hybrid ACT results separately labelled.
- [x] ACT trained/exported; hybrid fresh-seed test 19/20, matched normal/blank RGB 10/10 vs 1/10.
- [x] Retrieve fork/spoon from the open drawer without resetting between skills.
      Fork 10/10 on held-out seeds 40-49, chained straight after the drawer in ONE
      episode, no weld, no teleport, no reset (`outputs/fork-supportgate-40-49.json`).
- [x] Implement plate placement with BOTH arms: 10/10 held-out seeds 50-59, 9/10 on
      tuning seeds 40-49, open-gripper control 0/4. A single arm grips the rim 94 mm
      from the plate's centre of mass and only levers it, so the single-arm teacher is
      0/10 and is kept as the documented contrast, not deleted.
- [ ] Implement a measured giver/receiver handoff. Gates are wired (receiver lift is
      now required, not just contact duration) but no passing run exists yet.
- [ ] Chain the individual skills, track ownership, recover and evaluate the whole task.
- [ ] Replace unreliable VLM probe with tested scene/instruction reasoning.
- [ ] Validate simulation and inference on actual Core Ultra Series 2/3 hardware.

Public-code audit: `docs/PUBLIC_CODE_AUDIT.md`. The closest public full-task project
reports 0/10 complete sequences; two narrower 10/10 prototypes use weld/kinematic
attachment. We will not import those shortcuts or copy unlicensed competitor code.

The new drawer and existing cup run in separate reset workcells for now. They do
not yet satisfy bimanual coordination or the full dinner-table prompt. Detailed
milestone notes below include historical targets, not claims of completed work.

Source: uploaded Intel Physical AI Online Challenge, five pages, reviewed 2026-09-14.
Project: `/Users/beyond/Desktop/projects/bimanual-tabletop-vla`.
Local console: `http://127.0.0.1:8771`. Start with `uv run tabletop-web --port 8771`.

## Reality check

The earlier scaffold loaded two arms but did not execute tasks. A finite physics state
and two structural tests were not evidence of grasping, coordination, or reasoning.
The keyword planner previously placed a fork before requesting a handoff by an arm
that did not own it. This sequence is corrected, but the planner remains a constrained
preview. It does not understand arbitrary language, pronouns, explicit arm assignments,
or scene images. Never expose its output as executed task success.

The brief's technical objective 4 asks for training or fine-tuning a robotics policy;
therefore a classical controller is our teacher/baseline, not the final fulfillment of
the learned-policy requirement. The final demonstration section specifically requires
Core Ultra Series 2/3 even though the platform section uses broader wording.

## Architecture decision

Python/MuJoCo owns physics, state, rendering, motor limits, and evaluation. A loopback
HTTP console displays rendered frames and sends user control commands. Keeping MuJoCo
local avoids porting contact physics into a separate browser engine.

First make a physical pick/lift/place succeed for one small object. Then collect
successful demonstrations for a compact ACT policy. Add a local supported VLM for
instruction + raw camera -> validated skill intent; maintain explicit ownership and
postconditions in the executor. Keep ground-truth simulator state available to the
teacher and scorer, but isolate it from the camera-policy evaluation path.

The camera VLM does not run every physics tick. Run it at skill boundaries and after
failures; a policy runs at the control rate and physics at 500 Hz. OpenVINO conversion
and measured behavior parity apply to the actual model used by the demo.

## Requirements and current evidence

| Brief requirement | Current implementation | Remaining verification |
|---|---|---|
| Two SO-101 arms, MuJoCo | Official pinned models; 12 limited motors | Reach/workspace/collision analysis |
| Drawer, cutlery, plate, cup | Sliding tray, cabinet, graspable cutlery handle, rimmed plate, hollow cup, target markers | Spoon handoff; bottle is set dressing with no pour skill |
| Grasp/place, coordinated action | Cup, drawer, drawer+fork chain, and a TWO-ARM plate lift, all contact-gated with negative controls | Handoff and recovery |
| Language + raw camera reasoning | Camera feed; keyword plan preview | Actual VLM, grounding, state updates, replan |
| Policy training or fine-tuning | Official LeRobot ACT baseline trained on ten recorded episodes | Successful independent physical rollouts before deployment |
| Placement, weight, friction, shape, lighting, background perturbations | Placement/mass/friction/light + cup aspect ratio + table/floor colors | More shapes/topologies, clutter and broader workspace |
| Ten randomized seeds | Every skill measured on ten held-out seeds: drawer 10/10, fork 10/10, two-arm plate 10/10, camera cup 49/50 on 500-549 | Ten full dinner-table task rollouts |
| OpenVINO optimization | Validated-input benchmark with hashes/device/timing metadata | Real export, Intel run and behavior-parity checks |
| Core Ultra Series 2/3 demonstration | No qualifying hardware run | User-provided Intel host and device benchmark |
| Reproducible public repo + video | Local repository, lockfile, tests, provenance | Training/eval commands, video, user-approved publication |

## Sequenced tasks

### M0 — Inspectable local foundation

- [x] Preserve pinned official SO-101 assets and license.
- [x] Frame both arms and add overview/top cameras.
- [x] Add drawer slider, simple cutlery, and placement markers.
- [x] Provide browser play/pause, step, reset, joint control, plan preview.
- [x] Add deterministic placement/mass/friction/light randomization.
- [x] Add position-only IK that changes actuator targets, not live qpos.
- [x] Test reset repeatability, seed variation, ten-seed stability, actuator limits,
      physical response, invalid input, and reach behavior.
- [x] Correct the fork ownership error in the baseline plan.
- [x] Make physics independent of browser polling; automatic reconnect and HTTP integration test.
- [x] Open native MuJoCo viewer; add separate cup pickup and motor-demo controls.

Acceptance: browser displays a real rendered frame, stepping changes time, motor
targets cause measured joint movement, identical seeds reproduce object poses.
Commands: `uv run pytest -q`; `uv run tabletop-web --port 8771`.

### M1 — Reliable single-arm manipulation (depends on M0)

- [x] Replace solid cup with hollow compound rectangular cup (handle still pending).
- [x] Establish tool frame at contact gap; document convex finger collision proxies.
- [x] Add downward-orientation IK with residual/limit checks. Force saturation analysis remains.
- [ ] Map reachable object/target regions before selecting randomization ranges.
- [x] Implement approach, descend, close, lift, carry, release, retreat states.
- [x] Check force-bearing two-jaw contact, continuous lift, and stable released placement.
- [ ] Detect empty grasp, dropped object, timeout, force saturation, and blocked motion.
- [x] Add one bounded camera re-observe/retry and a separately labelled ACT hybrid
      retreat; neither uses a weld or object teleport.

Acceptance: one object's pick/lift/place succeeds repeatedly with independent physical
postconditions and failure controls (open gripper/no controller must fail). Log joint
targets vs actual positions and contact evidence. Proposed command: `tabletop-eval
--task mug-place --seeds 0:10` (not implemented yet).

### M2 — Bimanual dinner-table sequence (depends on M1)

- [x] Add a drawer housing and reachable handle workcell; open it by physical
      left-arm grip/pull (10/10 seeds 30–39, separate from the cup workcell).
- [ ] Route each arm through collision-checked waypoints with a shared-zone lock.
- [x] Calibrate the live jaw-face midpoint and closing axis; add object-specific
      opening, squeeze, and standoff parameters for fork, spoon, plate, and mug.
      `gripper_geometry()` reads the live jaw faces from the collision boxes and the
      teachers re-solve against the achieved midpoint.
- [x] Add closing-axis orientation to IK. `solve_pose(..., closing=)` constrains the
      jaw closing direction, which is what makes a radial rim pinch possible at all.
      NOT done: rejecting waypoints missed through joint/torque saturation despite a
      low IK residual, and `Solution` still reports no achieved-closing-axis error, so
      an accepted pose can have quietly abandoned the closing constraint it was given.
- [ ] Implement giver hold -> receiver contact -> giver release -> receiver lift handoff.
- [ ] Track object ownership and reject out-of-order or impossible skills. A handoff
      requires receiver bilateral contact, giver release, receiver lift, and sustained
      receiver ownership; sequential touches do not count.
- [x] Complete drawer, cutlery, plate, and cup placement as verified subgoals. All four
      have held-out seed rates and open-gripper controls that score 0. They are verified
      SUBGOALS; they are not yet chained into one episode beyond drawer -> fork.
- [ ] Add hold-and-pour only after handoff succeeds; explicitly specify a measurable
      pouring proxy if fluid dynamics are omitted.

Acceptance: full-sequence results separate actual placement, accidental proximity,
handoff, and partial completion. Both arms participate in a measured cooperative action.

### M3 — Camera reasoning and learned policy (depends on reliable M1 data)

- [x] Record RGB, proprioception, actions, timestamps, instruction, seed, and outcomes.
- [x] Split by episode and seed; reserve evaluation seeds outside training/validation.
- [x] Integrate official LeRobot 0.6.0 ACT; first training baseline saved.
- [x] Train ACT, compare teacher/PyTorch/OpenVINO/hybrid rollouts, and retain failed results.
- [ ] Integrate supported local VLM at skill boundaries with a strict output schema.
- [x] Reject unsupported intent, disallowed objects, invalid arm ownership, and malformed plans.
- [x] Reobserve once after a failed camera grasp; retain image/model probe evidence.
- [x] Compare camera behavior with the state teacher and a blank-image learned-policy baseline.

Acceptance: learned policy controls at least one skill; actual camera observations
influence action selection, verified by controlled perturbations and ablations.

### M4 — OpenVINO and Intel validation (depends on model selection, Intel access)

- [x] Export ACT to OpenVINO IR, retain FP32 baseline and precision metadata; reject
      the FP16-weight candidate because action drift was too large.
- [ ] Add calibration data for INT8 where supported and measure output/action drift.
- [x] Fix benchmark limitations: positive sample counts, real fixed-shape input fixture,
      warmup count, model checksum, OpenVINO version, full device name, precision,
      compile time, and batch-aware sequential throughput labeling.
- [x] Run matched PyTorch/OpenVINO seeds; retain the observed 8/10 versus 7/10
      closed-loop difference rather than claiming behavioral parity.
- [ ] Run simulation and inference on Core Ultra Series 2/3; record CPU/iGPU/NPU availability.

Acceptance: real hardware JSON and reproducible commands, no substitution of Mac or
other CPU numbers for the required Intel result. GPU/NPU support is measured per model.

### M5 — Evaluation and submission evidence (depends on M2–M4)

- [x] Add bounded cup aspect-ratio and table/floor color perturbations. Arbitrary
      object topology and cluttered backgrounds remain uncovered.
- [ ] Run ten held-out seeds; log subgoals, full success, recovery, time, and failures.
- [ ] Record command, seed variation, camera/policy inference, arm motion, and outcome.
- [ ] Verify clean checkout setup on the Intel host, preserve all upstream notices.
- [ ] Prepare technical summary and limitations from measured results.
- [ ] Recheck event deadline/resources via organizer channel; cached listing isn't a current clock.
- [ ] Publish/submit only with the user's explicit authorization.

## Decision checkpoints

1. Prioritize a physical grasp over richer UI or model downloads.
2. If contact mechanics block progress, reduce object/task complexity while documenting
   the deviation; do not silently change robot torque limits or scoring rules.
3. If no reliable teacher exists, do not spend the remaining window training from failed demos.
4. Confirm Intel hardware access before investing heavily in final optimization.

## Verified sources

- User-supplied PDF, pages 1–5: full technical scope, required deliverables, rubric.
- https://github.com/TheRobotStudio/SO-ARM100/tree/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101
- https://github.com/huggingface/lerobot
- https://github.com/openvinotoolkit/openvino_notebooks
- https://github.com/openvinotoolkit/openvino.genai
- https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html
- https://github.com/open-edge-platform/physical-ai-studio
- https://github.com/openvinotoolkit/physicalai

The initial resource search missed the official Intel hackathon resource page. It
links Physical AI Studio (training/export) and OpenVINO Physical AI (inference/runtime).
Both have now been cloned and inspected. Neither is a complete dual-SO101 dinner-table
scene/task solution. Local ACT uses the official LeRobot package; no other team's
task-controller code has been incorporated.

## Latest evidence (2026-09-14)

- Cup teacher: seeds 0–9 passed; separate 100–149 batch passed 50/50 with original
  four perturbations. Seeds 200–209 passed 10/10 with cup aspect-ratio and table/floor
  color perturbations added. These are cup-skill results, not full challenge scores.
- Open-gripper ablation: 0/10 successes, as intended; the lift gate detects failure.
- Recorded ten successful train episodes (1000–1009) and three validation episodes
  (1100–1102). Raw observations align with issued controls before physics integration.
- LeRobot ACT: 500 training steps on Apple MPS, best validation normalized L1 0.07464.
  Physical closed-loop acceptance remains separate; do not equate loss with success.
- No Intel machine has been verified. Mac training is permitted by the brief but
  cannot fulfill the final Intel simulation/inference demonstration.
