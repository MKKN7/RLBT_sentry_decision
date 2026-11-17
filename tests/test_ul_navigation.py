import numpy as np
import pytest

from rl_bt.ul_navigation import ULNavigation


@pytest.fixture(scope="module")
def nav():
    return ULNavigation()


def floor_square(z, size=4.):
    return np.asarray([[[0., 0., z], [size, 0., z], [size, size, z]],
                       [[0., 0., z], [size, size, z], [0., size, z]]])


def test_own_supply_and_center_connected_but_enemy_supply_footprint_forbidden(nav):
    for team, own, enemy, near_enemy in ((0, [1.5, 7.], [10.5, 1.], [9.3, 1.]),
                                         (1, [10.5, 1.], [1.5, 7.], [2.7, 7.])):
        assert nav.is_free(own, team)
        assert not nav.is_free(enemy, team)
        # Centre is outside enemy supply, but the body overlaps it.
        assert not nav.is_free(near_enemy, team)
        assert nav.plan(own, enemy, team) is None
        path = nav.plan(own, [6., 4.], team)
        assert path is not None
        np.testing.assert_allclose(path[0], own)
        np.testing.assert_allclose(path[-1], [6., 4.])
        for start, end in zip(path[:-1], path[1:]):
            # Check the actual polyline, including endpoint-to-centre segments.
            for alpha in np.linspace(0., 1., 9):
                assert nav.is_free(start + alpha * (end - start), team)


def test_reference_envelope_blocks_both_tunnels_highlands_and_workbenches(nav):
    assert nav.radius_m == pytest.approx(.411)
    assert nav.height_m == pytest.approx(.255)
    for xy in ([3.5, 7.46], [8.5, .54], [3.5, 4.8], [8.5, 3.2], [.4, 7.8], [11.6, .2]):
        for team in (0, 1):
            assert not nav.is_free(xy, team)
    for team in (0, 1):
        candidates = nav.goal_candidates(team)
        assert len(candidates) == 14
        assert [c["id"] for c in candidates] == list(range(14))
        assert all(c["static_legal"] for c in candidates[:10])
        assert all(not c["static_legal"] for c in candidates[10:])
        assert all(c["source"] == "engineering_anchor" for c in candidates[2:10])


def test_unknown_support_and_floor_bottom_are_not_ground():
    envelope = dict(planning_radius_m=.1, extra_planar_margin_m=0.,
                    planning_height_m=.2, extra_vertical_margin_m=0.)
    lower_only = ULNavigation(triangles=floor_square(-.1), envelope=envelope,
                              bounds_xy=((0., 0.), (4., 4.)))
    assert not lower_only.supported_cells.any()
    assert lower_only.plan([1., 1.], [3., 3.], 0) is None
    valid = ULNavigation(triangles=floor_square(0.), envelope=envelope,
                         bounds_xy=((0., 0.), (5., 5.)))
    assert valid.is_free([1., 1.], 0)
    assert not valid.is_free([4.5, 4.5], 0)
    assert not valid.is_free([.01, 2.], 0)


def test_vertical_wall_is_detected_even_though_height_rays_miss_it():
    # Wall spans the entire supported domain and separates its two sides.
    wall = np.asarray([[[2., 0., 0.], [2., 4., 0.], [2., 4., .4]],
                       [[2., 0., 0.], [2., 4., .4], [2., 0., .4]]])
    mesh = np.concatenate((floor_square(0.), wall))
    envelope = dict(planning_radius_m=.1, extra_planar_margin_m=0.,
                    planning_height_m=.2, extra_vertical_margin_m=0.)
    nav = ULNavigation(triangles=mesh, envelope=envelope, bounds_xy=((0., 0.), (4., 4.)))
    assert nav.is_free([1., 2.], 0) and nav.is_free([3., 2.], 0)
    assert not nav.is_free([2., 2.], 0)
    assert nav.plan([1., 2.], [3., 2.], 0) is None


def test_graph_cannot_cut_diagonal_corners(nav):
    free = np.asarray([[True, False], [False, True]])
    original_shape = nav.shape
    try:
        nav.shape = free.shape
        graph = nav._make_graph(free)
    finally:
        nav.shape = original_shape
    assert graph.nnz == 0


def test_invalid_or_blocked_start_and_goal_not_snapped_across_obstacles(nav):
    for bad in ([float("nan"), 1.], [float("inf"), 0.], [-.1, 4.], [12., 4.], [3.5, 4.8]):
        assert not nav.is_free(bad, 0)
        assert nav.plan(bad, [6., 4.], 0) is None
        assert nav.plan([6., 4.], bad, 0) is None
    with pytest.raises(ValueError):
        nav.plan([6., 4.], [6., 4.], 2)
    with pytest.raises(ValueError):
        nav.is_free([6., 4.], True)


def test_goal_cache_reused_and_no_hardware_certification(nav):
    start, goal = [1.5, 7.], [6., 4.]
    first = nav.plan(start, goal, 0)
    key = (0, nav._cell(goal)[0] * nav.shape[1] + nav._cell(goal)[1])
    tree = nav._goal_trees[key]
    np.testing.assert_array_equal(first, nav.plan(start, goal, 0))
    assert nav._goal_trees[key] is tree
    assert not nav.summary()["hardware_reachability_certified"]
    assert not nav.summary()["actual_Nav2_MPPI_connected"]
