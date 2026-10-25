from fire_sim.models import MonitoringPoint, changes


def point(i, risk=1):
    return MonitoringPoint(i, i[0] * 10 + 5, i[1] * 10 + 5, risk, 900 / risk)


def test_change_membership_and_attributes():
    before = {(0, 0): point((0, 0)), (1, 0): point((1, 0)), (2, 0): point((2, 0))}
    after = {(1, 0): point((1, 0)), (2, 0): point((2, 0), risk=3), (3, 0): point((3, 0))}
    delta = changes(before, after)
    assert delta.added == {(3, 0)} and delta.removed == {(0, 0)} and delta.changed == {(2, 0)}
    assert delta.affected_current == {(2, 0), (3, 0)}
    assert delta.fraction(before, after) == 3 / 4
    assert not changes(before, dict(before)).affected_current


def test_removal_only_has_empty_c_t():
    before = {(0, 0): point((0, 0))}
    delta = changes(before, {})
    assert delta.removed == {(0, 0)} and not delta.affected_current
    assert delta.fraction(before, {}) == 1
