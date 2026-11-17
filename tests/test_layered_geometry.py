import numpy as np
import pytest

from rl_bt.layered_geometry import rasterize_vertical_surfaces


def square(z, x0=0, x1=1):
    return np.asarray([[[x0, 0, z], [x1, 0, z], [x1, 1, z]],
                       [[x0, 0, z], [x1, 1, z], [x0, 1, z]]], dtype=float)


def test_lower_and_upper_surfaces_preserved_with_vertical_separation():
    mesh = np.concatenate((square(0), square(.3), square(.3)[..., ::-1, :]))
    result = rasterize_vertical_surfaces(mesh, resolution=.5)
    np.testing.assert_equal(result["surface_count_xy"], 2)
    np.testing.assert_allclose(result["surface_z_m"], [0, .3] * 4)
    np.testing.assert_allclose(result["z_to_next_surface_m"][::2], .3)
    assert np.isnan(result["z_to_next_surface_m"][1::2]).all()
    assert result["low_slope_surface_candidate"].all()


def test_slope_interpolates_height_and_winding_does_not_classify_ground():
    mesh = square(0)
    mesh[..., 2] = mesh[..., 0] * np.tan(np.radians(25))
    a = rasterize_vertical_surfaces(mesh, resolution=.5)
    b = rasterize_vertical_surfaces(mesh[:, ::-1], resolution=.5)
    np.testing.assert_allclose(a["surface_z_m"], b["surface_z_m"])
    np.testing.assert_allclose(a["min_incident_slope_deg"], 25)
    assert not a["low_slope_surface_candidate"].any()
    np.testing.assert_allclose(a["surface_z_m"], np.repeat([.25, .75], 2) * np.tan(np.radians(25)))


def test_no_triangle_means_no_hit_and_vertical_walls_are_not_free_cells():
    mesh = np.concatenate((square(0, 0, .4),
        np.asarray([[[.8, 0, 0], [.8, 1, 0], [.8, 1, 2]]])))
    result = rasterize_vertical_surfaces(mesh, resolution=.5, bounds_xy=[[0, 0], [1, 1]])
    np.testing.assert_equal(result["surface_count_xy"], [[1, 1], [0, 0]])
    assert int(result["vertical_or_degenerate_face_count"]) == 1
    assert result["column_offsets"].tolist() == [0, 1, 2, 2, 2]


def test_nearby_levels_not_merged_and_invalid_inputs_rejected():
    mesh = np.concatenate((square(0), square(.02)))
    result = rasterize_vertical_surfaces(mesh, resolution=.5, merge_tolerance_m=.001)
    assert len(result["surface_z_m"]) == 8
    with pytest.raises(ValueError):
        rasterize_vertical_surfaces(mesh, resolution=0)
    with pytest.raises(ValueError):
        rasterize_vertical_surfaces(mesh * np.nan)
