import pytest

from tabletop_vla.planning.task_planner import plan


def test_planner_emits_bounded_skill_schema():
    skills = plan("Open the drawer, place the plate and mug, then hand the fork to the left arm")
    assert skills[0].name == "open_drawer"
    assert any(skill.name == "handoff" for skill in skills)
    assert all(skill.arm for skill in skills)


def test_fork_handoff_preserves_giver_ownership():
    steps = [s for s in plan("Hand the fork to the left arm") if s.object == "fork"]
    assert [(s.name, s.arm) for s in steps] == [("pick", "right"), ("handoff", "right_to_left")]


def test_unsupported_negation_is_not_executed():
    with pytest.raises(ValueError):
        plan("Do not open the drawer")
