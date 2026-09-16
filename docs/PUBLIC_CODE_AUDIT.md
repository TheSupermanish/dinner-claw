# Public SO-101 and Intel Physical AI code audit

Checked 2026-09-14. This document records ideas and evidence, not code copied into
this repository. Reimplement useful mechanisms independently unless an upstream
dependency is deliberately adopted with its license and notice files.

## Closest challenge implementation

### jianwang-ntu/bimanual-dinner-table-so101 (MIT)

URL: https://github.com/jianwang-ntu/bimanual-dinner-table-so101

This is the closest public match to the uploaded task: two SO-101 arms, drawer,
fork, spoon, plate, mug, seeded evaluation, a scripted controller, learned-policy
experiments, and OpenVINO export. Its reporting is unusually candid: individual
subgoals sometimes pass, but the complete five-subgoal sequence is reported as
0/10. Its handoff/contact monitor can also count sequential touches too loosely,
so its handoff result must not be treated as proof of a receiver-held transfer.

Ideas worth independently implementing:

1. Compute the live midpoint between the jaw collision faces and the jaw-closing
   axis. Do not use a fixed wrist site as the grasp point.
2. Account for the SO-101's asymmetric gripper: one jaw moves and the other is
   stationary. Apply object-specific opening, squeeze, and standoff distances.
3. Add the closing-axis orientation to IK, not position alone, for cutlery and rims.
4. Route through measured cabinet-clearance waypoints and verify the achieved pose;
   a low IK residual does not prove that torque-limited servos reached it.
5. Track postconditions and object ownership independently of the controller.
6. For a real handoff require receiver bilateral contact, giver release, receiver
   lift, and sustained ownership. Sequential touches alone are insufficient.

## Useful upstream infrastructure

### openvinotoolkit/physicalai (Apache-2.0)

URL: https://github.com/openvinotoolkit/physicalai

Best source for the final inference contract: action chunks, OpenVINO/ONNX
backends, bimanual targets, device selection, and benchmark structure. Reuse its
runtime conventions where they fit, while keeping our task scorer independent.

### TheRobotStudio/SO-ARM100 SO101 model (Apache-2.0)

URL: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

Official SO-101 MuJoCo geometry and actuator baseline. This project already pins
and attributes these assets; preserve upstream limits and provenance.

### johannstark/so-101-mujoco (Apache-2.0)

URL: https://github.com/johannstark/so-101-mujoco

Useful later for a Gymnasium-compatible environment and MuJoCo Warp parallelism.
Parallel environments become valuable after a single full episode is correct; they
do not solve contact geometry or task validity.

### rocPAI-Forge/so101-simstudio (Apache-2.0 plus third-party notices)

URL: https://github.com/rocPAI-Forge/so101-simstudio

Strong record/replay/validate workflow for LeRobot v3 datasets, multiple cameras,
teleoperation, ACT, and SmolVLA. Its current supported training path is Ubuntu +
AMD ROCm, not macOS, so adopt the workflow ideas rather than replacing our working
Mac environment wholesale.

### imeun/SO101-MuJoCo-Depth-ACT (no detected repository license)

URL: https://github.com/imeun/SO101-MuJoCo-Depth-ACT

Useful research reference for dual-depth ACT, temporal action chunks, DAgger,
episode replay, joint mapping, and sim-to-real diagnostics. It reports a checkpoint
that passed 2/3 inspected MuJoCo episodes and says the real-robot learned policy is
not yet reliable. With no detected license, do not copy its code.

### ataghof/molmoact2-so101-sim (Apache-2.0)

URL: https://github.com/ataghof/molmoact2-so101-sim

Useful evidence that a large pretrained VLA is not a shortcut: the repository
reports zero-shot failure and materially better grasp than full-task success after
fine-tuning. It is also much heavier than needed for the current Mac-first build.

## Public challenge prototypes to study, not emulate blindly

### Ahmadbey678/intel-bimanual-vla-hackathon (no detected license)

URL: https://github.com/Ahmadbey678/intel-bimanual-vla-hackathon

Reports 10/10 for a narrow cup pick-handoff-place, but attaches the cup
kinematically, uses a fixed fallback plan in 7/10 reported episodes, has no learned
motor policy, and optimizes CLIP rather than its hosted Gemini planner. Its useful
idea is honest fallback logging; its score is not comparable to physical grasping.

### ashish-doing/bimanual-vla-manipulation (no detected license)

URL: https://github.com/ashish-doing/bimanual-vla-manipulation

Has a language planner, verify/replan loop, and dashboard, but its handoff uses
MuJoCo weld constraints and direct object reset logic. The UI/event-log pattern is
useful; the manipulation result is not evidence for the challenge's physical grasp.

## Decision for this project

Do not retrain the cup or import another repository. The fastest defensible path is:

1. Add jaw midpoint/closing-axis calibration and orientation-aware IK.
2. Build a fork teacher first, with a spoon variant sharing the same state machine.
3. Add achieved-waypoint, torque-saturation, bilateral-contact, lift, and drop gates.
4. Add a strict ownership state machine before implementing handoff.
5. Build one no-reset sequence: drawer -> fork retrieval -> transfer/place -> cup.
6. Record/replay only successful teacher episodes, then train a compact ACT skill.
7. Use OpenVINO Physical AI's action-chunk/runtime conventions for deployment.

The competitive opportunity is not to claim the widest feature set. It is to show a
smaller sequence with physical contact, negative controls, camera ablation, and ten
honestly scored seeds. Public code shows that reliable cutlery and strict handoff are
the unsolved parts across current entries.
