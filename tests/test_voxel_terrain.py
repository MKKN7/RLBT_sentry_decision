"""Terrain semantics: support, slopes, low roofs, steps, unknown and scan time."""
from dataclasses import replace

import numpy as np
import pytest

from rl_bt.voxel_terrain import VoxelTerrainConfig, VoxelTerrainMap


def cloud_plane(angle_deg=0., extent=1.5, z_offset=0.):
    xy = np.arange(-extent + .025, extent, .05)
    x, y = np.meshgrid(xy, xy, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(),
        (x * np.tan(np.deg2rad(angle_deg)) + z_offset).ravel()))


def terrain(points, *, config=None, pose=(0., 0., 0.)):
    config = config or VoxelTerrainConfig(resolution_m=.1, radius_m=2.,
        self_exclusion_radius_m=0., seed_height_tolerance_m=.025)
    mapper = VoxelTerrainMap(config)
    assert mapper.update(points, sampled_at_s=1., now_s=1.,
                         robot_position_xyz=pose, frame_id="odom") == "accepted"
    return mapper, mapper.snapshot(robot_position_xyz=pose, now_s=1.)


def value(result, xy):
    cell = np.floor((np.asarray(xy) - result.origin_xy_m) / result.resolution_m).astype(int)
    return result.occupancy_xy[tuple(cell)]


def test_floor_is_support_unobserved_cells_stay_unknown():
    _, out = terrain(cloud_plane())
    assert value(out, (.5, .5)) == 0
    assert value(out, (1.8, 1.8)) == -1
    assert not len(out.obstacle_points_xyz)
    assert out.diagnostics["simulator_truth_used"] is False
    assert out.diagnostics["unknown_is_free"] is False


def test_twenty_degree_ramp_inclusive_and_steep_ramp_blocked():
    _, good = terrain(cloud_plane(20.))
    _, steep = terrain(cloud_plane(25.))
    assert value(good, (.6, .2)) == 0
    assert good.slope_xy_deg[np.isfinite(good.slope_xy_deg)].max() == pytest.approx(20., abs=1e-4)
    assert value(steep, (.6, .2)) != 0
    assert steep.diagnostics["free_cells"] == 0


def test_support_classification_is_invariant_to_session_height_origin():
    points = cloud_plane(10.)
    _, original = terrain(points)
    shifted = points + np.array([0., 0., 2.37])
    _, translated = terrain(shifted, pose=(0., 0., 2.37))
    np.testing.assert_array_equal(original.occupancy_xy, translated.occupancy_xy)
    valid = np.isfinite(original.support_height_xy_m)
    # Absolute voxel bin boundaries change centroid grouping; millimetric
    # height quantization is allowed, but free/occupied/unknown labels agree.
    np.testing.assert_allclose(translated.support_height_xy_m[valid]-2.37,
                               original.support_height_xy_m[valid], atol=.005)


def test_wall_and_low_roof_are_obstacles_high_roof_is_not():
    floor = cloud_plane()
    y, z = np.meshgrid(np.arange(-1., 1., .05), np.arange(.05, 1., .04), indexing="ij")
    wall = np.column_stack((np.full(y.size, .8), y.ravel(), z.ravel()))
    _, with_wall = terrain(np.vstack((floor, wall)))
    assert value(with_wall, (.85, .25)) == 100
    low_roof = cloud_plane(z_offset=.23)
    _, low = terrain(np.vstack((floor, low_roof)))
    assert value(low, (.5, .5)) == 100
    assert low.diagnostics["separate_surface_layers"] > low.diagnostics["observed_columns"]
    _, high = terrain(np.vstack((floor, cloud_plane(z_offset=.4))))
    assert value(high, (.5, .5)) == 0


def test_unconnected_flat_deck_is_not_teleported_into_support():
    points = cloud_plane()
    points[points[:, 0] >= .4, 2] = .2
    _, out = terrain(points)
    assert value(out, (-.5, .2)) == 0
    assert value(out, (1., .2)) != 0


def test_height_uncertainty_does_not_seed_disconnected_40mm_step():
    points = cloud_plane()
    points[points[:, 0] >= .4, 2] = .04
    cfg = VoxelTerrainConfig(resolution_m=.1, radius_m=2.,
        self_exclusion_radius_m=0., seed_height_tolerance_m=.045)
    _, out = terrain(points, config=cfg)
    assert out.diagnostics["support_seeds"] == 1
    assert value(out, (1., .2)) != 0


def test_unknown_height_is_not_assumed_floor_and_sparse_scan_stays_unknown():
    _, out = terrain(cloud_plane(z_offset=.2))
    assert out.diagnostics["support_seeds"] == 0
    assert out.diagnostics["free_cells"] == 0
    _, sparse = terrain(np.array([[.5, .5, 0.], [.6, .5, 0.]]))
    assert sparse.diagnostics["free_cells"] == 0


def test_timestamp_frame_rejection_and_expiration_do_not_refresh_map():
    mapper, out = terrain(cloud_plane())
    points = cloud_plane(z_offset=.2)
    for expected, sample, now, frame in (
        ("frame_mismatch", 1.1, 1.1, "map"),
        ("future_sample", 2., 1.1, "odom"),
        ("stale_sample", .1, 1.1, "odom"),
        ("out_of_order", .9, 1.1, "odom"),
    ):
        assert mapper.update(points, sampled_at_s=sample, now_s=now,
            robot_position_xyz=(0., 0., 0.), frame_id=frame) == expected
    after = mapper.snapshot(robot_position_xyz=(0., 0., 0.), now_s=1.1)
    np.testing.assert_array_equal(out.occupancy_xy, after.occupancy_xy)
    expired = mapper.snapshot(robot_position_xyz=(0., 0., 0.), now_s=5.)
    assert expired.diagnostics["observed_voxels"] == 0
    assert np.all(expired.occupancy_xy == -1)


def test_contact_offset_is_explicit_and_ros_occupancy_order_is_x_fastest():
    cfg = VoxelTerrainConfig(resolution_m=.1, radius_m=2.,
        self_exclusion_radius_m=0., support_contact_offset_m=.5)
    _, out = terrain(cloud_plane(), config=cfg, pose=(0., 0., .5))
    assert value(out, (.5, .5)) == 0
    expected = out.occupancy_xy.T.ravel().tolist()
    assert out.ros_data() == expected


def test_rolling_retention_voxel_limit_and_finite_validation():
    cfg = VoxelTerrainConfig(radius_m=2., max_voxels=20, self_exclusion_radius_m=0.)
    mapper, _ = terrain(cloud_plane(), config=cfg)
    assert len(mapper._voxels) <= 20
    out = mapper.snapshot(robot_position_xyz=(10., 10., 0.), now_s=1.1)
    assert out.diagnostics["observed_voxels"] == 0
    with pytest.raises(ValueError, match="Finite estimated"):
        mapper.snapshot(robot_position_xyz=(np.nan, 0., 0.), now_s=1.)
    with pytest.raises(ValueError, match="Slope"):
        replace(cfg, max_slope_deg=90.)
