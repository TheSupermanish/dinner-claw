"""Deterministic instruction to skill-plan mapping, with ownership taken from measurement.

The plan is a preview. `executable_task` is the gate that decides what may actually drive
motors, and it fails closed. A language model may later emit this same schema, but it is
validated against the same measured ownership table before anything executes.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass

ALLOWED_SKILLS = {"open_drawer", "pick", "place", "handoff", "pour"}
# "both" is a real ownership value, not a placeholder. The plate is lifted by two arms
# gripping opposing rim segments; a single arm grips 94 mm from its centre of mass and
# only levers it, measured 0/10. A plan that assigns the plate to one arm is wrong and
# `validate` rejects it.
ARMS = {"left", "right", "both"}

# Arm ownership is DERIVED FROM MEASUREMENT, not from a naming convention. Every row cites
# the evidence that fixed it. An earlier version of this file assigned arms by alternating
# index, which put the drawer on the right arm and invented a right-to-left fork handoff;
# both are contradicted by the runs below.
OWNERSHIP = {
    # object:  (pick arm, place arm, evidence)
    "drawer": ("left", "left",
               "10/10 seeds 30-39, outputs/drawer-final-matmoved-30-39.json"),
    "fork": ("left", "left",
             "10/10 held-out seeds 40-49, outputs/fork-heldout-tuned-40-49.json"),
    "spoon": ("left", "right",
              ("left retrieves it (outputs/spoon-retrieve-v1-30-32.json) then is 244 mm "
               "short of spoon_mat; the right arm is 95.6 mm short of the drawer")),
    "mug": ("right", "right",
            "50/50 seeds 200-249, outputs/cup-six-perturbations-200-249.json"),
    "plate": ("both", "both",
              ("two-arm rim lift 10/10 on held-out seeds 50-59, 9/10 tuning 40-49, "
               "open-gripper control 0/4. A SINGLE arm is 0/10: it grips 94 mm from the "
               "plate's centre of mass, so it levers the plate onto its far rim and six "
               "of ten seeds never leave the table at all")),
    # No arm owns the bottle, and the row says so rather than inventing one. Measured
    # 2026-09-16 over tilt 0-90 by azimuth 0-360 at three heights: the LEFT arm is
    # 186.7 mm short, so the previous hardcoded `left` for pouring was contradicted by
    # the geometry. The right arm reaches the bottle's position to 1.2 mm but no
    # approach axis clears the orientation gate, so it has no accepted grasp pose
    # either. Pour stays previewable and non-executable.
    "bottle": ("right", "right",
               ("nearest arm only, reproduce with `python -m tabletop_vla.sim.reach "
                "--target bottle`: right reaches position to 0.0012 m but NO approach "
                "axis is accepted, left is 0.1867 m short, and the 70 mm body exceeds "
                "the 56.4 mm jaw opening. No grasp pose and no pour skill, so this is "
                "a plan row, not a capability")),
}

# Skills with a measured physical success rate on held-out seeds. Everything else may be
# previewed but must never be handed to the executor.
EXECUTABLE = {
    # The cup command resolves to the CAMERA teacher, not the scripted one; that is the
    # variant that demonstrates camera grounding and it is what the console has always run.
    "camera-cup-place": r"pick up (?:the )?(?:blue )?(?:cup|mug)(?: with (?:the )?right arm)?"
                 r" and place it (?:on|at) (?:its|the) (?:marker|mat|target)",
    "fork-retrieve": r"(?:open (?:the )?drawer and )?"
                     r"(?:take|get|pick up|retrieve) (?:the )?fork"
                     r"(?: from (?:the )?drawer)?"
                     r"(?: and (?:place|put) it (?:on|at) (?:its|the) (?:marker|mat|target))?",
    # The plate command resolves to the BIMANUAL teacher: opposing rim grips cancel
    # the pivot torque that keeps the single-arm plate at 0/10. The pattern requires
    # the literal plate, so it cannot steal cup/mug or fork instructions, and the
    # bare single-arm phrasing ("place the plate on its marker") still refuses.
    "bimanual-plate-lift": r"(?:pick up|lift) (?:the )?plate with both (?:arms|hands)"
                           r"|move (?:the )?plate to (?:its|the) (?:marker|mat|target)",
}


@dataclass(frozen=True)
class Skill:
    name: str
    arm: str
    object: str
    target: str | None = None


def plan(instruction: str) -> list[Skill]:
    """Deterministic, inspectable preview. Never evidence that anything executed."""
    text = re.sub(r"\bcup\b", "mug", instruction.lower())
    if not re.search(r"\b(open|place|pick|hand|handoff|pour|set|take|get|retrieve)\b", text):
        raise ValueError("Use an action such as open, place, pick, take, hand, or pour")
    if re.search(r"\b(not|don't|never|avoid)\b", text):
        raise ValueError("Negated instructions need the future language model; preview refused")
    skills: list[Skill] = []
    if "drawer" in text and re.search(r"\b(open|take|get|retrieve)\b", text):
        skills.append(Skill("open_drawer", OWNERSHIP["drawer"][0], "drawer"))
    for obj in ("plate", "mug", "fork", "spoon"):
        if obj not in text:
            continue
        picker, placer, _ = OWNERSHIP[obj]
        skills.append(Skill("pick", picker, obj))
        # The handoff is DERIVED: it exists exactly when the arm that can reach the object
        # is not the arm that can reach its target. It is never hardcoded per object.
        if picker != placer:
            skills.append(Skill("handoff", f"{picker}_to_{placer}", obj, "handoff_zone"))
        skills.append(Skill("place", placer, obj, f"{obj}_mat"))
    if "pour" in text:
        # Derive the arm from the measured table like every other object. Hardcoding
        # "left" here asserted an arm that cannot reach the bottle at all.
        picker, _, _ = OWNERSHIP["bottle"]
        skills.extend((Skill("pick", picker, "bottle"),
                       Skill("pour", picker, "bottle", "mug")))
    return validate(skills)


def executable_task(instruction: str) -> str:
    """Fail closed: never silently drop an unsupported portion of a command."""
    if not isinstance(instruction, str):
        raise TypeError("Instruction must be text")
    normalized = " ".join(instruction.lower().strip().rstrip(".! ").split())
    for task, pattern in EXECUTABLE.items():
        if re.fullmatch(pattern, normalized):
            return task
    raise ValueError(
        "Only the verified commands are executable, meaning ones with a measured physical "
        "success rate on held-out seeds: the cup pick-and-place, and fork retrieval from "
        "the drawer. Plate placement (0/10), the spoon handoff, pouring, negations and "
        "arbitrary instructions are not executable.")


def validate(skills: list[Skill]) -> list[Skill]:
    """Reject a plan that asks an arm to do something the measurements say it cannot."""
    if not skills:
        raise ValueError("Instruction did not map to any supported skill")
    for skill in skills:
        if skill.name not in ALLOWED_SKILLS:
            raise ValueError(f"Planner produced an unsupported skill: {skill.name}")
        if skill.name == "handoff":
            giver, _, receiver = skill.arm.partition("_to_")
            if giver not in ARMS or receiver not in ARMS or giver == receiver:
                raise ValueError(f"Handoff needs two different arms, got {skill.arm!r}")
            continue
        if skill.arm not in ARMS:
            raise ValueError(f"Unknown arm {skill.arm!r}")
        if skill.object in OWNERSHIP:
            picker, placer, evidence = OWNERSHIP[skill.object]
            expected = placer if skill.name == "place" else picker
            if skill.arm != expected:
                raise ValueError(
                    f"{skill.arm} arm cannot {skill.name} the {skill.object}: {evidence}")
    return skills


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    parser.add_argument("--executable", action="store_true",
                        help="Resolve to a runnable task name instead of a preview plan")
    args = parser.parse_args()
    if args.executable:
        print(executable_task(args.instruction))
        return
    print(json.dumps([asdict(skill) for skill in plan(args.instruction)], indent=2))


if __name__ == "__main__":
    main()
