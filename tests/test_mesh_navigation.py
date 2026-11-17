import math

import numpy as np
import pytest

from rl_bt.mesh_navigation import (MeshSurfaceQuery, ReferenceEnvelope,
                                   triangle_box_intersections)


def square(z, x0=0., x1=2., y0=0., y1=2.):
    return np.asarray([[[x0, y0, z], [x1, y0, z], [x1, y1, z]],
                       [[x0, y0, z], [x1, y1, z], [x0, y1, z]]])


def test_sat_rejects_triangle_whose_aabb_overlaps_box_without_triangle_overlap():
    # Triangle lies beyond x+y=1.2; its AABB still overlaps the +/-0.5 box.
    tri = np.asarray([[[.4, 1.2, 0], [1.2, .4, 0], [1.2, 1.2, 0]]])
    assert not triangle_box_intersections(tri, [.5, .5, .5])[0]
    assert triangle_box_intersections(tri * .5, [.5, .5, .5])[0]


def test_vertical_wall_and_roof_intersect_and_face_winding_is_irrelevant():
    wall = np.asarray([[[1., 0., 0.], [1., 2., 0.], [1., 2., 1.]],
                       [[1., 0., 0.], [1., 2., 1.], [1., 0., 1.]]])
    mesh = MeshSurfaceQuery(np.concatenate((square(0.), square(.25), wall)))
    envelope = ReferenceEnvelope(.4, .3, .20, .01)
    clear = mesh.probe_upright_pose(xy_m=[.5, 1.], support_z_m=0., yaw_rad=0.,
                                  envelope=envelope, support_tolerance_m=.001)
    assert clear["sampled_reference_pass"]
    blocked = mesh.probe_upright_pose(xy_m=[.95, 1.], support_z_m=0., yaw_rad=0.,
                                    envelope=envelope, support_tolerance_m=.001)
    assert blocked["reasons"] == ["body_box_intersects_mesh_surface"]
    tall = mesh.probe_upright_pose(xy_m=[.5, 1.], support_z_m=0., yaw_rad=0.,
        envelope=ReferenceEnvelope(.4, .3, .30, .01), support_tolerance_m=.001)
    assert "body_box_intersects_mesh_surface" in tall["reasons"]
    reversed_mesh = MeshSurfaceQuery(mesh.triangles[:, ::-1])
    np.testing.assert_equal(mesh.vertical_hits([.5, 1.]), reversed_mesh.vertical_hits([.5, 1.]))
    np.testing.assert_equal(mesh.intersecting_faces(center_xyz_m=[.5, 1., .2],
        half_extents_m=[.2, .15, .1]), reversed_mesh.intersecting_faces(
            center_xyz_m=[.5, 1., .2], half_extents_m=[.2, .15, .1]))


def test_layers_are_explicit_and_footprint_overhang_fails_support_samples():
    mesh = MeshSurfaceQuery(np.concatenate((square(0.), square(.3))))
    envelope = ReferenceEnvelope(.4, .4, .15, .01)
    options = dict(envelope=envelope, support_tolerance_m=.001, yaw_rad=0.)
    # Selecting the upper layer stays on it rather than aliasing to lower floor.
    assert mesh.probe_upright_pose(xy_m=[1., 1.], support_z_m=.3, **options)["sampled_reference_pass"]
    wrong = mesh.probe_upright_pose(xy_m=[1., 1.], support_z_m=.15, **options)
    assert not wrong["sampled_reference_pass"]
    assert "missing_or_wrong_layer_support_sample" in wrong["reasons"]
    edge = mesh.probe_upright_pose(xy_m=[.1, 1.], support_z_m=0., **options)
    assert not edge["sampled_reference_pass"]
    assert "missing_or_wrong_layer_support_sample" in edge["reasons"]


def test_obb_rotation_avoids_unrelated_world_aabb_surface():
    # Long box rotates along Y; obstacle at X=.8 no longer intersects it.
    obstacle = np.asarray([[[.8, -.2, -.2], [.8, .2, -.2], [.8, .2, .2]]])
    mesh = MeshSurfaceQuery(obstacle)
    assert len(mesh.intersecting_faces(center_xyz_m=[0, 0, 0], half_extents_m=[1, .1, .1]))
    assert not len(mesh.intersecting_faces(center_xyz_m=[0, 0, 0], half_extents_m=[1, .1, .1],
                                          yaw_rad=math.pi / 2))


def test_segment_samples_middle_obstacle_and_wraps_yaw():
    wall = np.asarray([[[1, 0, 0], [1, 2, 0], [1, 2, 1]],
                       [[1, 0, 0], [1, 2, 1], [1, 0, 1]]], dtype=float)
    mesh = MeshSurfaceQuery(np.concatenate((square(0.), wall)))
    result = mesh.sweep_upright_segment(start_xyzh=[.3, 1., 0., math.pi - .01],
        end_xyzh=[1.7, 1., 0., -math.pi + .01], envelope=ReferenceEnvelope(.2, .2, .2, .01),
        support_tolerance_m=.001, max_translation_step_m=.05, max_yaw_step_rad=.01)
    assert not result["sampled_reference_pass"]
    assert result["first_failure"]["reasons"] == ["body_box_intersects_mesh_surface"]
    assert result["actual_translation_step_m"] <= .05 + 1e-12
    assert result["actual_yaw_step_rad"] <= .01
    assert not result["hardware_reachability_certified"]


def test_input_validation_rejects_unknown_or_nonfinite_geometry():
    with pytest.raises(ValueError):
        ReferenceEnvelope(.4, .3, .2, 0.)
    with pytest.raises(ValueError):
        MeshSurfaceQuery(np.zeros((0, 3, 3)))
    mesh = MeshSurfaceQuery(square(0.))
    with pytest.raises(ValueError):
        mesh.probe_upright_pose(xy_m=[1., 1.], support_z_m=float("nan"), yaw_rad=0.,
            envelope=ReferenceEnvelope(.2, .2, .2, .01), support_tolerance_m=.001)
