GOAL
Expose the new `bimanual-plate-lift` skill through the runtime, the planner, and both demo
surfaces, so the thing that actually works is the thing a demo can run.

FACTS ALREADY ESTABLISHED (do not re-derive, do not re-measure)

The skill exists and is committed at `src/tabletop_vla/sim/bimanual_plate.py` (commit 4304e5d).
Its public names are exactly:

    from tabletop_vla.sim.bimanual_plate import BimanualPlateLift, prepare_bimanual_workcell

Constructor signature matches every other teacher: `BimanualPlateLift(sim, close_gripper=True)`.
Workcell signature matches too: `prepare_bimanual_workcell(sim, seed)`.
`state()["name"]` returns the string `"bimanual-plate-lift"`.

Measured results, already in the commit message. Quote these, never invent others:
  tuning seeds 40-49    9/10
  held-out seeds 50-59 10/10,  placement error 13.6-26.9 mm
  open-gripper control  0/4

The four wiring points, each with the line to copy the shape from:

1. `src/tabletop_vla/sim/runtime.py:113-146` `start_task`. The name whitelist is the set literal
   at :114-117. The dispatch is the `elif` chain; `plate-place-left` at :138-142 is the closest
   analogue, since it is the only other one that calls its own workcell function.
2. `src/tabletop_vla/planning/task_planner.py:40-49` `EXECUTABLE`. Entries are
   `"task-name": r"<regex>"`. Add a pattern matching natural phrasings for lifting the plate with
   both arms, e.g. "pick up the plate with both arms", "lift the plate with both hands",
   "move the plate to its mat". Follow the existing style: non-capturing optional groups.
3. `src/tabletop_vla/native.py:47-55`. The key set at :47 and the name map at :52-55. Use key `B`.
4. `src/tabletop_vla/console.js:146-150` for the button handler, plus the matching button in
   `src/tabletop_vla/console.html`. Copy the shape of the `plate-place-left` handler at :150.

WHAT THIS RULE DOES NOT APPLY TO

- Do NOT remove, rename, or downgrade `plate-place-left`. The single-arm plate is 0/10 and is
  kept deliberately as a documented failure; `AGENTS.md` forbids deleting failed results. It
  stays wired exactly as it is. You are ADDING a skill, not replacing one.
- Do NOT touch `src/tabletop_vla/sim/bimanual_plate.py` or `tests/test_bimanual_plate.py`.
  Another reviewer owns those files this round. If you believe the skill itself has a bug,
  report it in FINDINGS and change nothing.
- Do NOT change any existing regex in `EXECUTABLE`. `tests/test_planner.py` pins their behaviour
  and `camera-cup-place` in particular is load-bearing.
- Do NOT edit any build-progress copy, README, tasks.md, or AGENTS.md. Those are held by another
  session with uncommitted changes.

BLAST RADIUS

`EXECUTABLE` is consumed by `plan()` at `task_planner.py:90`. A new pattern that is too greedy
will steal instructions from `camera-cup-place` or `fork-retrieve`. That is the specific way this
change can be wrong, and a passing new test will not catch it, so the acceptance below re-runs
the WHOLE planner suite, not just a new case.

FILES YOU OWN
  src/tabletop_vla/sim/runtime.py
  src/tabletop_vla/planning/task_planner.py
  src/tabletop_vla/native.py
  src/tabletop_vla/console.js
  src/tabletop_vla/console.html
  tests/test_planner.py        (add cases; do not weaken existing ones)
Everything else is off limits.

ACCEPTANCE, each item a command whose OUTPUT is the evidence

  .venv/bin/python -m ruff check src tests                       -> exit 0
  node --check src/tabletop_vla/console.js                       -> exit 0
  .venv/bin/python -m pytest -q                                  -> 77 passed or more, 0 failed
  .venv/bin/python -c "
from tabletop_vla.sim.runtime import Simulation
s = Simulation()
try:
    print(s.start_task(41, 'bimanual-plate-lift')['name'])
finally:
    s.close()"                                                    -> prints bimanual-plate-lift

  .venv/bin/python -c "
from tabletop_vla.planning.task_planner import plan
for text in ('pick up the plate with both arms',
             'pick up the blue cup with the right arm and place it on its marker',
             'open the drawer and take the fork'):
    print(repr(text), [sk.name for sk in plan(text)])"
    -> the plate line resolves to the bimanual skill, and the OTHER TWO are unchanged.
       Paste all three lines of output into your report. If any existing one changed,
       the verdict is FAIL.

Do NOT pipe a command whose exit code you need through head or tail.

PERMISSION TO FAIL
If a claim cannot be measured, NAME it unmeasured. An honest "not measured, here is why" is worth
more than a fabricated confirmation. Do not run the 10-seed evaluations; they take many minutes
and their numbers are given above.

OUTPUT CONTRACT. Write your FULL report to `.agent-logs/wire-bimanual.md`. Print to stdout ONLY:

  VERDICT: PASS | FAIL | PARTIAL
  DID: <one line each>
  FINDINGS: <path:line: problem, or NONE>
  GATES: <command -> exit code, one per line>
  UNCHECKED: <one line each, or NONE>
  REPORT: <path>

If PASS needs a caveat, the verdict is PARTIAL.
