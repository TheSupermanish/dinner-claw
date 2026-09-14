from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass

ALLOWED_SKILLS = {"open_drawer", "pick", "place", "handoff", "pour"}


@dataclass(frozen=True)
class Skill:
    name: str
    arm: str
    object: str
    target: str | None = None


def plan(instruction: str) -> list[Skill]:
    """Deterministic, inspectable baseline; a VLM can later emit this same schema."""
    text = re.sub(r"\bcup\b", "mug", instruction.lower())
    if not re.search(r"\b(open|place|pick|hand|handoff|pour|set)\b", text):
        raise ValueError("Use an action such as open, place, pick, hand, or pour")
    if re.search(r"\b(not|don't|never|avoid)\b", text):
        raise ValueError("Negated instructions need the future language model; preview refused")
    skills: list[Skill] = []
    if "drawer" in text and "open" in text:
        skills.append(Skill("open_drawer", "right", "drawer"))
    for index, obj in enumerate(("plate", "mug", "fork", "spoon")):
        if obj not in text or (obj == "fork" and "hand" in text):
            continue
        arm = "left" if index % 2 == 0 else "right"
        skills.extend((Skill("pick", arm, obj), Skill("place", arm, obj, f"{obj}_mat")))
    if "hand" in text and "fork" in text:
        skills.append(Skill("pick", "right", "fork"))
        skills.append(Skill("handoff", "right_to_left", "fork", "handoff_zone"))
    if "pour" in text:
        skills.extend((Skill("pick", "left", "bottle"), Skill("pour", "left", "bottle", "mug")))
    return validate(skills)


def executable_task(instruction: str) -> str:
    """Fail closed: do not silently drop unsupported portions of a command."""
    if not isinstance(instruction, str):
        raise TypeError("Instruction must be text")
    normalized = " ".join(instruction.lower().strip().rstrip(".! ").split())
    pattern = (r"pick up (?:the )?(?:blue )?(?:cup|mug)"
               r"(?: with (?:the )?right arm)? and place it (?:on|at) (?:its|the) "
               r"(?:marker|mat|target)")
    if not re.fullmatch(pattern, normalized):
        raise ValueError("Only the verified command is executable: 'Pick up the blue cup and "
                         "place it on its marker.' Drawer, other objects, left-arm assignments, "
                         "pouring and arbitrary instructions are not executable yet.")
    return "camera-cup-place"


def validate(skills: list[Skill]) -> list[Skill]:
    if not skills:
        raise ValueError("Instruction did not map to any supported skill")
    if any(skill.name not in ALLOWED_SKILLS for skill in skills):
        raise ValueError("Planner produced an unsupported skill")
    return skills


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction")
    args = parser.parse_args()
    print(json.dumps([asdict(skill) for skill in plan(args.instruction)], indent=2))


if __name__ == "__main__":
    main()
