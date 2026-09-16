# Architecture, evidence, and implementation decisions

Updated 2026-09-14. Read this with `README.md`, `tasks.md`, and
`docs/PUBLIC_CODE_AUDIT.md` before changing the project.

## What is running now

| Layer | Current implementation | Evidence boundary |
|---|---|---|
| Physics | Local MuJoCo with two pinned SO-101 arms | Real contact simulation; browser and native viewers are separate instances |
| Language planning | Deterministic grammar/keyword preview | Not an LLM and not executable except for the one verified cup command |
| Visual grounding | Calibrated blue-cup color detector | Camera-dependent but limited to one known object/workcell |
| VLM probe | Pinned `HuggingFaceTB/SmolVLM-500M-Instruct` | Advisory only; saved probes were unreliable and cannot command motors |
| Motor policy | LeRobot 0.6.0 ACT | Learned single-cup controller; pure OpenVINO result 7/10 on its retained seed set |
| Deployment | ACT FP32 OpenVINO IR | Mac-tested; FP16 candidate rejected; no qualifying Intel Core Ultra result |
| Physical teachers | Cup place and drawer open | Verified separately; not yet a no-reset dinner-table sequence |

No LLM currently controls the robot. ACT is a visuomotor policy, not a language
model. The live console must continue to say this plainly.

## Target control loop

```text
instruction + overview image
            |
  local VLM at skill boundary
            |
 validated JSON skill plan
            |
 ownership + shared-zone checks
            |
 scripted/ACT skill controller
            |
       MuJoCo physics
            |
 independent postcondition scorer
            |
 bounded re-observe/replan or stop
```

The VLM never emits joint targets. It may only select from skills that already have
physical teachers, scorers, timeouts, and failure controls. Unsupported instructions
fail closed. Simulator ground truth is allowed for teacher generation and independent
scoring, never as a learned-policy observation.

## VLM decision

The current SmolVLM-500M probe is retained as negative evidence, not promoted. The
next candidate is `Qwen/Qwen2.5-VL-3B-Instruct`, evaluated first on a small frozen
suite of scene images and instructions. Intel maintains an OpenVINO notebook for this
model family:

https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen2.5-vl

Adoption gates:

1. Produce schema-valid JSON for every test case without silently dropping clauses.
2. Distinguish supported from unsupported objects/actions and respect explicit arms.
3. Use the image measurably: image swaps or blank frames must change/fail grounding.
4. Never execute a skill whose preconditions or ownership state are false.
5. Measure latency and output agreement after OpenVINO conversion on Intel hardware.

A hosted Groq/Gemini planner may be used only as a separately labelled development
comparison. It is not the preferred final path because the requirement emphasizes
local Intel/OpenVINO execution and external quotas can force unrepresentative fallbacks.

## Public-code verification performed locally

### `jianwang-ntu/bimanual-dinner-table-so101`

- All Python sources compiled.
- Scorer positive/negative controls passed 11/11.
- Scene verification passed 15/16; the sole failure was offscreen export because the
  research environment did not have optional `imageio`, not a scene/physics failure.
- A fresh headless scripted seed ran successfully as a program and achieved 1/5
  subgoals, 0/1 complete tasks. This agrees with its published 0/10 full-task result.
- Its handoff control proves the detector fires for sequential arm contacts; that is
  exactly why our stricter receiver-contact/release/lift/ownership gate is necessary.

Conclusion: working and valuable research code, but not a solved dinner-table task.

### `Ahmadbey678/intel-bimanual-vla-hackathon`

- All Python sources compiled and its MuJoCo quickstart loaded, randomized, rendered,
  and reported success on this Mac.
- Full pipeline was not rerun because its extra API/runtime dependencies are absent.
- Its documented 10/10 uses kinematic object attachment and mostly fixed planner
  fallbacks, so that number is not comparable with our force-bearing grasp criterion.

Conclusion: working narrow prototype and useful reporting reference, not physical
grasp/handoff evidence.

### `ashish-doing/bimanual-vla-manipulation`

- All Python sources compiled.
- Its own setup checker ran and reported missing `mink`, `dm_control`, FastAPI,
  Uvicorn, WebSockets, Groq, and `GROQ_API_KEY`; no dependencies were installed into
  our active project merely to reproduce it.
- Code inspection confirms a Groq `openai/gpt-oss-20b` JSON planner, but manipulation
  uses weld constraints/direct reset mechanics.

Conclusion: useful planner/schema/dashboard patterns; not reusable physical evidence.

## Improvements adopted independently

1. `gripper_geometry()` now measures the current inner jaw faces, midpoint, closing
   axis, and separation from our own collision pads. Tests prove the midpoint moves
   with the asymmetric jaw and the opening exceeds the closed separation.
2. The next cutlery IK will target that live midpoint and constrain the closing axis.
3. Handoff scoring will require receiver bilateral contact, giver release, receiver
   lift, and sustained ownership—not merely touches by both arms.
4. Every controller must verify the achieved waypoint and detect saturation/blocking.
5. Public repositories without a detected license remain study-only; no competitor
   implementation code is copied into this project.

## Next verified build slice

1. Add fork/spoon dimensions and object-specific jaw opening/standoff configuration.
2. Implement an already-open-drawer fork teacher using the live jaw geometry.
3. Add empty-grasp, collision, saturation, sustained-lift, and drop failure reports.
4. Run positive and open-gripper controls over seeds before recording demonstrations.
5. Reuse the same state machine for the spoon only after the fork passes.
6. Then implement ownership and the real transfer sequence.
