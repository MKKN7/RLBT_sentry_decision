#!/usr/bin/env python3
"""Read-only identity/coordinate audit of the existing UC simulator resources.

This never runs the source project's scripts or changes its configuration.  It
does not infer competition semantics from CAD coordinates and does not claim a
contact model or tunnel traversal has been physically validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_bt.layered_geometry import read_binary_stl
from rl_bt.mesh_navigation import MeshSurfaceQuery, ReferenceEnvelope


CONTAINER_PROBE = r'''
import hashlib, json, pathlib, struct
import numpy as np
import yaml

base = pathlib.Path('/opt/mid360_mujoco/generated/rmuc2027-v5')
manifest_path = base / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3,3)), ('attribute', '<u2')])

def inspect_stl(path):
    path = pathlib.Path(path)
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError('Binary STL structure differs: '+str(path))
    triangles = np.frombuffer(data, dtype=dtype, offset=84)['vertices'].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError('Nonfinite vertices: '+str(path))
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'triangles': count, 'bounds_m': [triangles.min(axis=(0,1)).tolist(),
                                             triangles.max(axis=(0,1)).tolist()]}, triangles

prepared_info, prepared = inspect_stl(base / 'field.stl')
source_info, source = inspect_stl(manifest['source'])
translation = np.asarray(manifest['translation'], dtype=float)
if prepared.shape != source.shape:
    raise ValueError('Prepared/source triangle shapes differ')
delta = prepared - (source + translation)
max_error = float(np.abs(delta).max())
same_surface = max_error < 3e-6
original_path = pathlib.Path('/opt/pb2025_nav_ws/src/pb2025_light_sim/meshes/rmuc_2027.stl')
original_info, original = inspect_stl(original_path) if original_path.exists() else (None, None)
source_difference = None
if original is not None:
    # Exact float32 vertex triplets, with per-triangle vertex order ignored.
    # This counts geometric records, not nearest-neighbour guesses.
    def triangle_keys(triangles):
        values = np.ascontiguousarray(triangles, dtype='<f4')
        rows = values.view('V12').reshape(-1,3)
        return np.ascontiguousarray(np.sort(rows, axis=1)).view('V36').reshape(-1)
    original_keys, driveable_keys = triangle_keys(original), triangle_keys(source)
    original_only = ~np.isin(original_keys, driveable_keys)
    driveable_only = ~np.isin(driveable_keys, original_keys)
    def changed_bounds(triangles, selection):
        selected = triangles[selection]
        return None if not len(selected) else [selected.min(axis=(0,1)).tolist(), selected.max(axis=(0,1)).tolist()]
    source_difference = {
        'comparison': 'exact float32 vertex triangle records ignoring vertex ordering',
        'original_triangles_absent_from_driveable': int(original_only.sum()),
        'driveable_triangles_absent_from_original': int(driveable_only.sum()),
        'original_only_bounds_source_frame_m': changed_bounds(original, original_only),
        'driveable_only_bounds_source_frame_m': changed_bounds(source, driveable_only),
        'geometrically_identical': not original_only.any() and not driveable_only.any(),
    }

pieces = manifest['pieces']
files = sorted((base / 'collision').glob('*.obj'))
listed = {p['file'] for p in pieces}
actual = {p.name for p in files}
if listed != actual:
    raise ValueError('Collision manifest/files mismatch')
piece_digest = hashlib.sha256()
for p in files:
    piece_digest.update(p.name.encode()+b'\0')
    piece_digest.update(hashlib.sha256(p.read_bytes()).digest())

robot_path = pathlib.Path('/opt/mid360_mujoco/configs/robot.yaml')
robot = yaml.safe_load(robot_path.read_text())
spawn = robot['spawn']
# Exact vertical triangle intersections at the original simulator's open test
# position.  Multiple surfaces are retained; no lowest-height collapse.
xy = np.asarray(spawn[:2], dtype=float)
a, b, c = prepared[:,0,:2], prepared[:,1,:2], prepared[:,2,:2]
v0, v1, v2 = b-a, c-a, xy-a
denom = v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0]
valid = np.abs(denom) > 1e-12
u = np.zeros(len(denom)); v = np.zeros(len(denom))
u[valid] = (v2[valid,0]*v1[valid,1]-v2[valid,1]*v1[valid,0])/denom[valid]
v[valid] = (v0[valid,0]*v2[valid,1]-v0[valid,1]*v2[valid,0])/denom[valid]
inside = valid & (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
zs = prepared[:,0,2] + u*(prepared[:,1,2]-prepared[:,0,2]) + v*(prepared[:,2,2]-prepared[:,0,2])
surfaces = np.unique(np.round(zs[inside], 7)).tolist()

result = {
    'container': 'Combat_Sentry2026',
    'prepared_field_dir': str(base),
    'prepared_mesh': prepared_info,
    'driveable_source_mesh': source_info,
    'original_unsuffixed_rmuc_mesh': original_info,
    'original_vs_driveable_difference': source_difference,
    'source_matches_manifest_sha256': source_info['sha256'] == manifest['source_sha256'],
    'translation_applied_once_m': translation.tolist(),
    'prepared_vs_translated_source_max_vertex_error_m': max_error,
    'prepared_preserves_source_triangle_order_and_surface_within_3um': same_surface,
    'field_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'collision': {
        'count': len(files), 'matches_manifest_file_set': True,
        'aggregate_sorted_name_and_sha256': piece_digest.hexdigest(),
        'obj_bytes': sum(p.stat().st_size for p in files),
        'model_builder_uses_piece_positions': True,
        'limitations_from_source_manifest': manifest['limitations'],
    },
    'visual_mesh_count': len(list((base/'visual').glob('*.stl'))),
    'source_robot_config': {'path': str(robot_path),
        'sha256': hashlib.sha256(robot_path.read_bytes()).hexdigest(),
        'open_test_spawn_xyz_yaw': spawn,
        'spawn_is_rule_defined': False,
        'vertical_surface_intersections_at_spawn_xy_m': surfaces},
    'read_only': True,
    'original_files_modified': False,
    'full_contact_or_tunnel_traversal_validated': False,
    'official_referee_coordinate_registration_validated': False,
    'rfid_and_healing_semantics_inferred_from_mesh': False,
}
print(json.dumps(result, indent=2))
'''


def local_routes(report):
    """Finite geometric route probes, never live observations or action masks."""
    mesh = ROOT / "assets/navigation_reference/field/field.stl"
    cache_path = ROOT / "data/derived/uc_surface_layers_010m/surfaces.npz"
    cache_report_path = cache_path.with_name("report.json")
    cached_metadata = json.loads(cache_report_path.read_text())
    if cached_metadata["source_sha256"] != report["prepared_mesh"]["sha256"]:
        raise ValueError("Surface cache does not match the checked prepared mesh")
    cache = np.load(cache_path)
    all_triangles = read_binary_stl(mesh)
    # Restrict the exact query to a padded local AABB, retaining every triangle
    # that intersects it; this is only a broad phase, not mesh simplification.
    xy_lo, xy_hi = np.array([1.8, 8.9]), np.array([5.2, 10.5])
    keep = (np.all(all_triangles.max(axis=1)[:, :2] >= xy_lo, axis=1)
            & np.all(all_triangles.min(axis=1)[:, :2] <= xy_hi, axis=1))
    query = MeshSurfaceQuery(all_triangles[keep])
    start_xy = np.array(report["source_robot_config"]["open_test_spawn_xyz_yaw"][:2])
    initial_surfaces = query.vertical_hits(start_xy)
    expected_support_z = .2607622
    start_support_z = float(initial_surfaces[np.argmin(np.abs(
        initial_surfaces[:, 0] - expected_support_z)), 0])
    robot = json.loads((ROOT / "configs/reference_robot_envelope.json").read_text())
    planning_radius = robot["planning_radius_m"] + robot["extra_planar_margin_m"]
    total_height = robot["planning_height_m"] + robot["extra_vertical_margin_m"]
    # An axis-aligned square enclosing the full planning circle.  The 15 mm
    # lower gap avoids counting the gentle tilted floor as a body wall; tyre
    # contacts and suspension must subsequently be checked by MuJoCo.
    lower_gap = .015
    envelope = ReferenceEnvelope(2*planning_radius, 2*planning_radius,
                                 total_height-lower_gap, lower_gap)
    routes = []
    for name, delta in (("forward_1p2m", [1.2, 0.]),
                        ("backward_1m", [-1., 0.]),
                        ("diagonal_0p8_0p4m", [.8, .4])):
        end_xy = start_xy + delta
        intervals = int(np.ceil(np.linalg.norm(delta)/.05))
        samples, heights, slopes = [], [], []
        for xy in np.linspace(start_xy, end_xy, intervals+1):
            hits = query.vertical_hits(xy)
            valid = hits[(np.abs(hits[:, 0]-start_support_z) <= .04)
                         & (hits[:, 1] <= 20.+1e-6)]
            if not len(valid):
                raise RuntimeError("No same-level route support at "+str(xy))
            support = valid[np.argmin(np.abs(valid[:, 0]-start_support_z))]
            heights.append(float(support[0])); slopes.append(float(support[1]))
            pose = query.probe_upright_pose(xy_m=xy, support_z_m=float(support[0]),
                yaw_rad=0., envelope=envelope, support_tolerance_m=.02,
                max_slope_deg=20., support_stencil=3)
            index = np.floor((xy-cache["origin_xy_m"])/float(cache["resolution_m"])).astype(int)
            shape = cache["grid_shape_xy"]
            flat = int(index[0]*shape[1]+index[1])
            lo, hi = cache["column_offsets"][flat:flat+2]
            z = cache["surface_z_m"][lo:hi]
            cached_height = float(z[np.argmin(np.abs(z-support[0]))])
            higher = hits[hits[:, 0] > support[0]+1e-5]
            samples.append({
                "world_xy_m": xy.tolist(),
                "local_xy_m": (xy-start_xy).tolist(),
                "selected_support_z_m": float(support[0]),
                "selected_support_slope_deg": float(support[1]),
                "all_vertical_intersections_z_slope": hits.tolist(),
                "nearest_cache_support_z_m": cached_height,
                "cache_support_error_m": abs(cached_height-float(support[0])),
                "next_higher_intersection_gap_m": None if not len(higher)
                    else float(higher[0, 0]-support[0]),
                "no_higher_intersection_means": "unknown_or_open_not_certified_clearance",
                "sampled_support_stencil_pass": all(p["supported"] for p in pose["support_samples"]),
                "body_box_intersecting_face_count": len(pose["intersecting_face_ids"]),
                "sampled_reference_pass": pose["sampled_reference_pass"],
            })
        route = {
            "id": name, "start_world_xy_m": start_xy.tolist(),
            "goal_world_xy_m": end_xy.tolist(), "goal_local_xy_m": delta,
            "length_m": float(np.linalg.norm(delta)),
            "support_height_range_m": [min(heights), max(heights)],
            "support_height_change_m": heights[-1]-heights[0],
            "max_selected_slope_deg": max(slopes),
            "sample_count": len(samples), "support_samples_per_pose": 9,
            "maximum_translation_sampling_step_m": .05,
            "max_cache_support_error_m": max(s["cache_support_error_m"] for s in samples),
            "sampled_reference_pass": all(s["sampled_reference_pass"] for s in samples),
            "dynamic_contact_validated": False, "samples": samples,
        }
        if not route["sampled_reference_pass"]:
            raise RuntimeError("Proposed local reference route is blocked: "+name)
        routes.append(route)
    return {
        "scope": "finite_offline_surface_and_box_probes_for_engineering_route_selection",
        "mesh_source": str(mesh.relative_to(ROOT)),
        "source_is_driveable_derivative": True,
        "surface_cache": str(cache_path.relative_to(ROOT)),
        "surface_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "full_mesh_triangles": len(all_triangles), "local_aabb_query_triangles": int(keep.sum()),
        "planning_circle_radius_m": planning_radius,
        "conservative_square_envelope_length_width_m": [envelope.length_m, envelope.width_m],
        "body_query_vertical_interval_above_center_support_m": [lower_gap, total_height],
        "support_tolerance_m": .02,
        "world_from_local_xy_translation_m": start_xy.tolist(),
        "world_from_local_yaw_rad": 0.,
        "registration_scope": "explicit_static_test_spawn_XY_only_no_dynamic_GT_injection",
        "official_map_and_referee_registration_verified": False,
        "recommended_first_goal_local_xy_m": routes[0]["goal_local_xy_m"],
        "authoritative_RL_action_mask": False,
        "hardware_or_continuous_sweep_certified": False,
        "routes": routes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="Combat_Sentry2026")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, default=ROOT/"configs/uc_map_reference.json")
    args = parser.parse_args()
    result = subprocess.run([
        "docker", "exec", args.container, "/opt/mid360_mujoco/.venv/bin/python",
        "-c", CONTAINER_PROBE,
    ], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    report["container"] = args.container
    snapshot_path = ROOT / "assets/navigation_reference/field/field.stl"
    data = snapshot_path.read_bytes()
    report["project_prepared_snapshot"] = {
        "path": str(snapshot_path), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "matches_live_prepared_mesh": hashlib.sha256(data).hexdigest() == report["prepared_mesh"]["sha256"],
    }
    readme_path = Path("/home/kellen/pb2025_sim_nuc/README.md")
    if readme_path.is_file():
        readme = readme_path.read_text().splitlines()
        evidence = [{"line": number, "text": line} for number, line in enumerate(readme, 1)
                    if "2026-11-03按用户要求生成" in line
                    or "原始来源为用户2026-11-02提供" in line
                    or "静态collision使用已整平" in line]
        report["source_documented_geometry_derivation"] = {
            "path": str(readme_path),
            "sha256": hashlib.sha256(readme_path.read_bytes()).hexdigest(),
            "evidence": evidence,
            "classification": "driveable_is_a_previous_flattened_derivative_not_the_original_rmuc_stl",
            "instructions_in_evidence_executed": False,
        }
    if not all((report["project_prepared_snapshot"]["matches_live_prepared_mesh"],
                report["source_matches_manifest_sha256"],
                report["prepared_preserves_source_triangle_order_and_surface_within_3um"])):
        raise RuntimeError("UC source identity/transform verification failed")
    report["status"] = "PASS_READ_ONLY_RESOURCE_IDENTITY"
    report["local_route_checks"] = local_routes(report)
    config = {
        "schema": "uc_selected_simulator_map_reference_v1",
        "user_selected_uc_geometry": True,
        "prepared_field_container_dir": report["prepared_field_dir"],
        "prepared_mesh_project_file": "assets/navigation_reference/field/field.stl",
        "prepared_mesh_sha256": report["prepared_mesh"]["sha256"],
        "geometry_is_previous_driveable_derivative": True,
        "source_mesh_already_translated": True,
        "additional_geometry_translation_or_scaling": False,
        "static_test_spawn_world_xyz_yaw": report["source_robot_config"]["open_test_spawn_xyz_yaw"],
        "static_test_spawn_is_rule_defined": False,
        "test_world_from_local_xy_translation_m": report["local_route_checks"]["world_from_local_xy_translation_m"],
        "test_world_from_local_yaw_rad": 0.,
        "test_world_from_local_z_offset_m": None,
        "test_registration_dynamically_updated_from_GT": False,
        "official_referee_coordinate_registration_verified": False,
        "recommended_first_goal_local_xy_m": [1.2, 0.],
        "local_engineering_routes": [{k:v for k,v in r.items() if k != "samples"}
            for r in report["local_route_checks"]["routes"]],
        "zone_semantics": {},
        "mesh_queries_are_live_actor_observations": False,
        "dynamic_contact_or_Nav2_result_in_this_config": False,
        "audit_report": str(args.out.resolve().relative_to(ROOT)),
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"report": str(args.out), "config": str(args.config_out),
        "status": report["status"],
        "first_goal_local_xy_m": config["recommended_first_goal_local_xy_m"],
        "sampled_route_passes": {r["id"]:r["sampled_reference_pass"]
            for r in report["local_route_checks"]["routes"]},
        "dynamic_contact_validated": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
