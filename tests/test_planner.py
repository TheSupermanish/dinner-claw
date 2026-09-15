import pytest

from tabletop_vla.planning.task_planner import (
    OWNERSHIP,
    Skill,
    executable_task,
    plan,
    validate,
)


def test_planner_emits_bounded_skill_schema():
    skills = plan("Open the drawer, place the plate and mug, then hand the spoon over")
    assert skills[0].name == "open_drawer"
    assert all(skill.arm for skill in skills)
    assert all(skill.name in {"open_drawer", "pick", "place", "handoff"} for skill in skills)


def test_drawer_is_owned_by_the_left_arm_that_actually_opens_it():
    """Measured 10/10 on seeds 30-39 with the LEFT arm. The previous planner said right."""
    (drawer,) = [s for s in plan("Open the drawer") if s.name == "open_drawer"]
    assert drawer.arm == "left"


def test_the_spoon_needs_a_left_to_right_handoff_and_the_fork_does_not():
    """The handoff is derived from measured reach, not hardcoded per object.

    The left arm retrieves the spoon from the drawer and is then 244 mm short of
    spoon_mat, while the right arm is 95.6 mm short of the drawer, so the spoon must
    change hands. The fork is left-arm end to end and scores 10/10 held out, so emitting
    a handoff for it would be inventing work the robot does not need.
    """
    spoon = [(s.name, s.arm) for s in plan("Place the spoon on its marker")]
    assert spoon == [("pick", "left"), ("handoff", "left_to_right"), ("place", "right")]
    fork = [(s.name, s.arm) for s in plan("Place the fork on its marker")]
    assert fork == [("pick", "left"), ("place", "left")]
    assert not any(s.name == "handoff" for s in plan("Place the fork on its marker"))


def test_every_ownership_row_cites_its_evidence():
    for obj, (picker, placer, evidence) in OWNERSHIP.items():
        assert picker in {"left", "right"}, obj
        assert placer in {"left", "right"}, obj
        assert "outputs/" in evidence or "seeds" in evidence, obj


def test_validate_rejects_an_arm_that_cannot_do_the_job():
    with pytest.raises(ValueError, match="cannot pick"):
        validate([Skill("pick", "right", "fork")])


def test_validate_rejects_a_handoff_between_one_arm_and_itself():
    with pytest.raises(ValueError, match="two different arms"):
        validate([Skill("handoff", "left_to_left", "spoon", "handoff_zone")])


def test_unsupported_negation_is_not_executed():
    with pytest.raises(ValueError):
        plan("Do not open the drawer")


def test_fork_retrieval_is_now_executable_because_it_was_measured():
    """10/10 on held-out seeds 40-49, outputs/fork-heldout-tuned-40-49.json."""
    assert executable_task("Open the drawer and take the fork") == "fork-retrieve"
    assert executable_task("Retrieve the fork from the drawer") == "fork-retrieve"


def test_bimanual_plate_lift_is_executable_while_single_arm_plate_is_not():
    """Tuning seeds 40-49 9/10, held-out seeds 50-59 10/10, open-gripper control 0/4."""
    assert executable_task("pick up the plate with both arms") == "bimanual-plate-lift"
    assert executable_task("lift the plate with both hands") == "bimanual-plate-lift"
    assert executable_task("move the plate to its mat") == "bimanual-plate-lift"
    with pytest.raises(ValueError, match="Only the verified"):
        executable_task("Place the plate on its marker")


def test_bimanual_plate_pattern_does_not_steal_cup_or_fork():
    assert executable_task(
        "pick up the blue cup with the right arm and place it on its marker"
    ) == "camera-cup-place"
    assert executable_task("open the drawer and take the fork") == "fork-retrieve"


def test_plate_and_spoon_remain_unexecutable_until_they_are_measured():
    for instruction in ("Place the plate on its marker",
                        "Hand the spoon to the right arm",
                        "Pour the bottle into the cup"):
        with pytest.raises(ValueError, match="Only the verified"):
            executable_task(instruction)
