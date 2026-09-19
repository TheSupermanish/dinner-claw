GOAL
Rewrite `tests/test_bimanual_plate.py` so that every test FAILS when the production line it
claims to cover is broken. Four of its five tests currently survive mutation, which means they
measure nothing.

THIS IS A FIX FOR A REVIEW FINDING, NOT A NEW TASK

An adversarial review (Codex, 2026-09-15) ran mutations against the production file and recorded
exactly which tests stayed GREEN while production was broken. Reproduce each mutation, confirm
the test still passes, then rewrite the test so it fails. These are the four, verbatim:

  test_open_gripper_control_never_scores_a_lift
      survives: disable the close-phase contact rejection
  test_lift_requires_all_four_pads_not_one_arms_two
      survives: accept ANY required body instead of all of them
  test_a_tipped_plate_cannot_score_a_lift_however_high_its_centre_rises
      survives: remove the tilt condition entirely
  test_retract_aims_from_the_live_jaw_position_not_the_nominal_mat
      survives: make production `enter()` use nominal-mat retract targets

Two mutations did NOT survive, so those tests are already real and their shape is worth copying:
  deleting the workcell placement -> its test failed correctly
  replacing the live position INSIDE `retract_candidates()` -> its test failed correctly

The review also named WHY each is weak:
  `tests/test_bimanual_plate.py:82`  asserts set membership without ever invoking the production
      contact gate. It checks `task.all_pads` as data; it never calls `after_step`.
  `tests/test_bimanual_plate.py:109` the tipped fixture is elevated but has no four-pad contact,
      so the lift accumulator was already going to refuse it for a DIFFERENT reason. Removing the
      tilt condition changes nothing. The fixture must satisfy every other condition and fail on
      tilt ALONE.
  `tests/test_bimanual_plate.py:130` tests the helper `retract_candidates()` while production
      `enter()` can bypass it. Cover the path `enter()` actually takes.
  `tests/test_bimanual_plate.py:59`  the open-gripper control at seed 41 touches only the right
      fixed pad at about 2.86 N, so it never exercises the four-contact false-positive case.

WHAT CHANGED IN PRODUCTION SINCE THE REVIEW (read the file; do not assume the old shape)

I have already fixed the four production defects. `src/tabletop_vla/sim/bimanual_plate.py` now:
  - `contacts()` returns GEOM ids, not body ids. `self.all_pads` is the four pad GEOMS
    (`left_fixed_pad`, `left_moving_pad`, `right_fixed_pad`, `right_moving_pad`), because the
    `left_gripper` BODY also owns `left_palm` and `left_fixed_neck`.
  - `tilt_deg()` no longer takes `abs()`, so an inverted plate reports 180.0 degrees.
  - the support test is a WHITELIST, `geoms <= self.all_pads`, not `0 not in bodies`.
  - `peak_tilt_deg` is measured over every airborne sample and is a SUCCESS CONDITION.

FILES YOU OWN
  tests/test_bimanual_plate.py
Everything else is off limits. In particular do NOT edit
`src/tabletop_vla/sim/bimanual_plate.py`: mutate it temporarily to prove a test bites, then
restore it with `git checkout -- src/tabletop_vla/sim/bimanual_plate.py`. Never restore by
moving a .bak file. Finish with `git diff --quiet -- src/tabletop_vla/sim/bimanual_plate.py`
returning 0.

WHAT THIS RULE DOES NOT APPLY TO
- Do not weaken or delete the two tests that already bite (workcell placement, live retract
  position). Strengthening them is fine; removing coverage is not.
- Do not add a test that needs a 10-seed evaluation. Every test must run in seconds. The
  existing suite completes in about 19 seconds and that is the budget.
- Do not assert on specific success COUNTS (9/10, 10/10). Those are measured separately and
  change when gates change. Test the gate logic, not the score.

ACCEPTANCE, each item a command whose OUTPUT is the evidence

  .venv/bin/python -m ruff check tests                     -> exit 0
  .venv/bin/python -m pytest tests/test_bimanual_plate.py -q  -> all pass, under 60s

  Then, for EACH of the four mutations above, in this exact loop:
    1. apply the mutation to src/tabletop_vla/sim/bimanual_plate.py
    2. .venv/bin/python -m pytest tests/test_bimanual_plate.py -q   -> MUST now FAIL,
       and report WHICH test failed
    3. git checkout -- src/tabletop_vla/sim/bimanual_plate.py
  Report all four as a table: mutation -> test that caught it. Any mutation with no catcher
  means the verdict is FAIL.

  git diff --quiet -- src/tabletop_vla/sim/bimanual_plate.py   -> exit 0

Do NOT pipe a command whose exit code you need through head or tail.

PERMISSION TO FAIL
If a mutation cannot be caught by a fast test, NAME it uncaught and say why. An honest "not
measured, here is why" is worth more than a fabricated confirmation. Do not fabricate a passing
mutation table.

OUTPUT CONTRACT. Write your FULL report to `.agent-logs/harden-bimanual-tests.md`. Print to
stdout ONLY:

  VERDICT: PASS | FAIL | PARTIAL
  DID: <one line each>
  MUTATION_TABLE: <mutation -> test that caught it, one per line>
  FINDINGS: <path:line: problem, or NONE>
  GATES: <command -> exit code, one per line>
  UNCHECKED: <one line each, or NONE>
  REPORT: <path>

If PASS needs a caveat, the verdict is PARTIAL.
