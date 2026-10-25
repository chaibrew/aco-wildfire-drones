from dataclasses import replace

import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from fire_sim.grid import Grid, changes, perimeter_changes
from fire_sim.scenarios import perturb


def test_boundary_only_including_both_sides_of_grid_lines():
    points = Grid(cell_size_m=10).points(box(0, 0, 30, 30))
    assert (1, 1) not in points
    assert (-1, 1) in points and (0, 1) in points
    assert (3, 1) in points and (2, 1) in points


def test_ids_and_risk_survive_growth():
    grid = Grid(cell_size_m=10)
    before = grid.points(box(-3, -3, 13, 13))
    after = grid.points(box(-3, -3, 23, 13))
    assert before.keys() & after.keys()
    for i in before.keys() & after.keys():
        assert before[i] == after[i]


def test_holes_and_multipolygons():
    shape = Polygon([(1, 1), (99, 1), (99, 99), (1, 99)],
                    holes=[[(31, 31), (69, 31), (69, 69), (31, 69)]])
    points = Grid(cell_size_m=10).points(MultiPolygon([shape, box(121, 1, 139, 19)]))
    assert (3, 4) in points
    assert (4, 4) not in points
    assert (12, 0) in points


def test_change_membership_attributes_and_local_geometry():
    grid = Grid(cell_size_m=10)
    before, after = box(1, 1, 9, 9), box(2, 2, 8, 8)
    a, b = grid.points(before), grid.points(after)
    assert not changes(a, b).changed
    assert perimeter_changes(grid, before, after, a, b).changed == {(0, 0)}
    b[(0, 0)] = replace(b[(0, 0)], revisit_s=123)
    assert changes(a, b).affected_current == {(0, 0)}
    delta = changes(a, {})
    assert delta.removed == {(0, 0)} and not delta.affected_current
    assert delta.fraction(a, {}) == 1


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_bad_grid_size(value):
    with pytest.raises(ValueError):
        Grid(cell_size_m=value)


def test_growth_is_repeatable_and_locality_changes_geometry():
    base = box(0, 0, 1000, 1000)
    local = perturb(base, radius_m=100, seed=9)
    assert local.equals(perturb(base, radius_m=100, seed=9))
    assert local.covers(base) and local.area > base.area
    assert not local.equals(perturb(base, radius_m=100, seed=9, locality="distributed"))
