"""Support projection rejects unknown terrain over the complete footprint."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from rl_bt.support_costmap import (FixedStartSupportPrior, SupportCostmapProjector,
    path_support_check, project_support_costmap)
from rl_bt.voxel_terrain import TerrainGrid


def grid(values, *, origin=(0., 0.), resolution=.1):
    values = np.asarray(values, dtype=np.int8)
    return TerrainGrid("odom", 1., np.asarray(origin, float), resolution, values,
        np.full(values.shape, np.nan), np.full(values.shape, np.nan),
        np.empty((0, 3)), {})


def test_projection_preserves_semantics_and_strictly_blocks_unknown():
    source = grid([[-1, 0, 100], [0, -1, 100]])
    result = project_support_costmap(source)
    np.testing.assert_array_equal(result.occupancy_xy, source.occupancy_xy)
    np.testing.assert_array_equal(result.strict_occupancy_xy, [[100, 0, 100], [0, 100, 100]])
    assert result.ros_data() == [-1, 0, 0, -1, 100, 100]
    assert result.ros_data(strict=True) == [100, 0, 0, 100, 100, 100]
    assert result.report["initial_support_prior"] is None
    assert not result.report["unknown_is_free"]
    result.occupancy_xy[0, 0] = 0
    assert source.occupancy_xy[0, 0] == -1


def test_fixed_start_prior_is_explicit_and_never_erases_obstacles():
    values = np.full((15, 15), -1, np.int8)
    values[7, 7] = 100
    prior = FixedStartSupportPrior((.75, .75))
    result = project_support_costmap(grid(values), initial_support_prior=prior)
    assert result.occupancy_xy[7, 7] == 100
    assert result.occupancy_xy[0, 0] == -1
    # Circle intersects this square although the square's center lies outside.
    assert np.linalg.norm(np.array([1.15, .95]) - [.75, .75]) > .411
    assert result.occupancy_xy[11, 9] == 0
    report = result.report["initial_support_prior"]
    assert report["unknown_cells_assumed_supported"] == int((result.occupancy_xy == 0).sum())
    assert report["occupied_cells_preserved"] == 1
    assert report["scan_observation"] is False
    assert report["fixed_in_session_frame"] is True


def test_session_prior_cannot_move_and_does_not_expand_with_rolling_grid():
    prior = FixedStartSupportPrior((.5, .5), radius_m=.2)
    projector = SupportCostmapProjector(prior)
    with pytest.raises(FrozenInstanceError):
        prior.center_xy_m = (1., 1.)
    with pytest.raises(FrozenInstanceError):
        projector.initial_support_prior = FixedStartSupportPrior((1., 1.))
    first = projector.project(grid(np.full((20, 20), -1)))
    rolled = projector.project(grid(np.full((20, 20), -1), origin=(1., 0.)))
    assert first.report["initial_support_prior"]["unknown_cells_assumed_supported"] > 0
    assert rolled.report["initial_support_prior"]["unknown_cells_assumed_supported"] == 0
    assert np.all(rolled.strict_occupancy_xy == 100)


def test_clear_continuous_path_passes_and_unknown_edge_of_footprint_fails():
    values = np.zeros((40, 40), np.int8)
    source = grid(values)
    assert path_support_check(source, [[1., 1.], [2., 1.]])["accepted"]
    values[15, 14] = -1  # Cell starts .4m above centerline; radius is .411m.
    result = path_support_check(grid(values), [[1., 1.], [2., 1.]])
    assert not result["accepted"]
    assert result["reason"] == "unknown_support"
    assert result["failure_cell_xy"] == [15, 14]
    assert 1. < result["failure_position_xy_m"][0] < 1.5


def test_sparse_waypoints_cannot_jump_over_one_cell_hole_or_obstacle():
    values = np.zeros((50, 30), np.int8)
    values[25, 15] = -1
    path = [[1., 1.55], [4., 1.55]]
    result = path_support_check(grid(values), path, radius_m=.05)
    assert result["reason"] == "unknown_support"
    assert result["failure_position_xy_m"][0] == pytest.approx(2.45)
    values[25, 15] = 100
    assert path_support_check(grid(values), path, radius_m=.05)["reason"] == "occupied"


def test_corner_intersection_and_tangency_are_conservative():
    values = np.zeros((30, 30), np.int8)
    values[15, 15] = -1
    # Corner of cell (1.5,1.5) is .1414m away; center is .2121m away.
    assert path_support_check(grid(values), [[1.4, 1.4]], radius_m=.15)["reason"] == "unknown_support"
    assert path_support_check(grid(values), [[1.4, 1.4]], radius_m=.14)["accepted"]
    assert path_support_check(grid(values), [[1.4, 1.55]], radius_m=.1)["reason"] == "unknown_support"


def test_path_reports_first_blocking_segment_and_contact_not_last_waypoint():
    values = np.zeros((60, 30), np.int8)
    values[35, 10] = 100
    values[45, 10] = -1
    result = path_support_check(grid(values), [[1., 1.05], [2., 1.05], [5., 1.05]], radius_m=.1)
    assert result["failure_segment_index"] == 1
    assert result["reason"] == "occupied"
    assert result["failure_position_xy_m"] == pytest.approx([3.4, 1.05])
    assert result["failure_segment_fraction"] == pytest.approx(1.4 / 3.)


def test_outside_map_rejected_even_when_all_visible_cells_are_free():
    source = grid(np.zeros((30, 30), np.int8))
    result = path_support_check(source, [[1., 1.], [2.9, 1.]])
    assert result["reason"] == "outside_map"
    assert result["failure_position_xy_m"] == pytest.approx([2.589, 1.])
    assert path_support_check(source, [[.1, 1.]])["reason"] == "outside_map"
    assert path_support_check(source, [[1., 1.], [1e100, 1.]])["reason"] == "outside_map"


def test_projection_and_path_validation_reject_invalid_inputs():
    source = grid(np.zeros((10, 10)))
    for path in ([], [[np.nan, 1.]], [[1., 2., 3.]]):
        with pytest.raises(ValueError, match="path"):
            path_support_check(source, path)
    with pytest.raises(ValueError, match="radius"):
        path_support_check(source, [[.5, .5]], radius_m=-1.)
    with pytest.raises(ValueError, match="only"):
        project_support_costmap(replace(source, occupancy_xy=np.ones((2, 2))))
    with pytest.raises(TypeError, match="Explicit"):
        project_support_costmap(source, initial_support_prior={"center_xy_m": [0., 0.]})
    with pytest.raises(ValueError, match="Finite"):
        FixedStartSupportPrior((np.inf, 0.))
