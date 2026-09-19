GOAL
Break the new two-arm plate skill. Find every way its 10/10 held-out result could be true on
paper while the robot did not do the work.

WHY YOU SPECIFICALLY
This project's dominant defect is a success gate that is NECESSARY but not SUFFICIENT. It has
shipped three times already, each time found only after the number had been reported:

  1. A fork scored as "retrieved" while resting on the OPEN DRAWER at z 0.79, because the
     placement gate checked xy, contact and velocity but not resting HEIGHT. Reported 8/10,
     retracted to 0/5.
  2. A handoff gate computed `receiver_lifts` from a 20 mm rise and then never read it, so
     bilateral contact duration alone counted as a transfer.
  3. The single-arm plate reported "sustained lift 2.4-7.9 s on all ten seeds" while the plate
     was TIPPING on its far rim, still resting on the table, because the lift gate checked the
     centre's height and not whether anything else bore the weight. 1/10 became a true 0/10.

The code under review was written to fix exactly that class of defect, by the same author who
shipped all three. Assume it has a fourth.

WHAT TO REVIEW (read-only; change nothing)
  src/tabletop_vla/sim/bimanual_plate.py
  tests/test_bimanual_plate.py
  git show 4304e5d --stat

THE CLAIMS, each of which you should try to falsify

  C1. "All four pads bearing force" proves two arms are gripping.
      `after_step` computes `held = self.all_pads <= bodies` from `contacts()`, which collects
      bodies in force-bearing contact with the plate (normal force > 0.02 N). Can all four pad
      bodies touch the plate without the plate being GRIPPED? Note the plate has 12 raised rim
      segments; on the SINGLE-arm skill an open gripper straddling a rim segment registered
      force-bearing contact on both of its pads. Why would four pads be different, and is the
      open-gripper control (0/4, fails at the four-pad gate with 0.00 s) actually evidence, or
      does it fail for an unrelated reason such as the jaws never reaching the rim at all?
  C2. "Tilt < 15 degrees" proves the plate is level.
      `tilt_deg()` uses `abs(zaxis[2])`. What does that do to a plate flipped upside down?
  C3. "The jaws are the only force-bearing contact" proves a lift.
      `unsupported = held and 0 not in bodies`. Body 0 is the world. Is every static support in
      this scene actually welded into body 0, or can the plate rest on something with its own
      body id (the drawer, the cabinet, another object) and still pass?
  C4. `sustained_lift` is a genuine continuous interval.
      Read the `lift_run` / `sustained_lift` accumulation. Can the reported number be assembled
      from separate short intervals, and does `tilt_at_peak_lift_deg` correspond to the interval
      actually reported?
  C5. The retract cannot drag the placed plate.
      `retract_candidates()` reads the LIVE plate position each call. `enter()` tries five
      clearance heights. Does trying five candidates run the IK against a plate position that
      has moved between candidates, and does the chosen candidate match the one reported?
  C6. The tests test the production line.
      MUTATION TEST, the decisive check. For each of the five tests, delete or invert the
      production line it claims to cover, re-run `.venv/bin/python -m pytest
      tests/test_bimanual_plate.py -q`, and report every test that stays GREEN. Restore with
      `git checkout -- <path>` and never by moving a .bak file. A test that passes against
      mutated production code is measuring nothing.

SPECIFICALLY DO NOT
- Do not re-run the 10-seed evaluations. They take many minutes. The reported numbers are
  tuning 9/10, held-out 10/10, control 0/4, and you are reviewing whether the GATES behind those
  numbers mean what they say, not reproducing them.
- Do not report style, naming, type hints, or docstring length. Only defects that could make a
  reported number wrong.
- Do not edit any file. If you mutate for the mutation test, restore with `git checkout --` and
  confirm `git status --short` is clean before finishing.

DISTINGUISH CONFIRMED FROM SUSPECTED. A CONFIRMED finding names the exact input or state that
produces the wrong result. Say explicitly what you did NOT check.

PERMISSION TO FAIL
If a claim cannot be measured, NAME it unmeasured. An honest "not measured, here is why" is worth
more than a fabricated confirmation. "I found nothing" is an acceptable verdict if it is true and
you say what you exercised to reach it.

OUTPUT CONTRACT. Write your FULL report to `.agent-logs/review-bimanual.md`. Print to stdout ONLY:

  VERDICT: PASS | FAIL | PARTIAL
  DID: <one line each>
  FINDINGS: <path:line: CONFIRMED|SUSPECTED: problem>
  MUTATIONS_SURVIVED: <test name -> mutated line, or NONE>
  GATES: <command -> exit code, one per line>
  UNCHECKED: <one line each, or NONE>
  REPORT: <path>

If PASS needs a caveat, the verdict is PARTIAL.
