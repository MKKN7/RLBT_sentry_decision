#!/usr/bin/env python3
"""Exercise UL reference mesh surface queries with an explicit synthetic box.

This does not run community scripts, ROS, Nav2 or MuJoCo.  Dimensions must be
provided on the command line so that no test box becomes an implicit hardware
calibration.  Results describe the frozen reference STL only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_bt.layered_geometry import read_binary_stl
from rl_bt.mesh_navigation import MeshSurfaceQuery, ReferenceEnvelope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-m", type=float, required=True)
    parser.add_argument("--width-m", type=float, required=True)
    parser.add_argument("--body-height-m", type=float, required=True)
    parser.add_argument("--belly-clearance-m", type=float, required=True)
    parser.add_argument("--support-tolerance-m", type=float, required=True)
    parser.add_argument("--translation-step-m", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest_path = ROOT / "assets/rmul_2027/derived/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mesh_path = ROOT / manifest["geometry_file"]
    mesh_sha = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if mesh_sha != manifest["geometry_sha256"]:
        raise ValueError("Mesh hash differs from recorded UL source manifest")
    query = MeshSurfaceQuery(read_binary_stl(mesh_path))
    envelope = ReferenceEnvelope(args.length_m, args.width_m,
                                 args.body_height_m, args.belly_clearance_m)
    tall_envelope = ReferenceEnvelope(args.length_m, args.width_m,
        max(.30, args.body_height_m), args.belly_clearance_m)
    cases = [
        ("red_supply_reference_surface", [1.5, 7.], .003, 0., envelope),
        ("open_floor", [6., 1.], 0., 0., envelope),
        ("red_highland_control_reference_surface", [3.5, 4.8], .2012, 0., envelope),
        ("red_tunnel_lower_layer", [3.5, 7.46], .0005, 0., envelope),
        ("blue_tunnel_lower_layer", [8.5, .54], .0005, 0., envelope),
        ("red_tunnel_taller_counterexample", [3.5, 7.46], .0005, 0., tall_envelope),
        ("red_tunnel_side_wall_counterexample", [3.5, 7.12], .0005, 0., envelope),
        ("wrong_support_layer_counterexample", [3.5, 7.46], .12, 0., envelope),
    ]
    probes = [{"id": name, **query.probe_upright_pose(xy_m=xy, support_z_m=z,
               yaw_rad=yaw, envelope=body, support_tolerance_m=args.support_tolerance_m)}
              for name, xy, z, yaw, body in cases]
    paths = []
    for name, start, end in [
        ("red_tunnel_crossing_reference", [2.8, 7.46, .0005, 0.], [4.2, 7.46, .0005, 0.]),
        ("blue_tunnel_crossing_reference", [9.2, .54, .0005, 0.], [7.8, .54, .0005, 0.]),
    ]:
        paths.append({"id": name, "start_xy_supportz_yaw": start,
            "end_xy_supportz_yaw": end, **query.sweep_upright_segment(
                start_xyzh=start, end_xyzh=end, envelope=envelope,
                support_tolerance_m=args.support_tolerance_m,
                max_translation_step_m=args.translation_step_m,
                max_yaw_step_rad=.05)})
    report = {"artifact_type": "offline_UL_mesh_surface_envelope_probe_not_navigation_acceptance",
        "mesh_file": str(mesh_path.relative_to(ROOT)), "mesh_sha256": mesh_sha,
        "manifest_file": str(manifest_path.relative_to(ROOT)),
        "mesh_triangles": len(query.triangles), "synthetic_envelope_cli": vars(args) | {"out": str(args.out)},
        "poses": probes, "sampled_segments": paths,
        "actual_Nav2_MPPI_connected": False, "hardware_reachability_certified": False,
        "may_be_used_as_authoritative_RL_action_mask": False,
        "limitations": [
            "The provided box is a synthetic upright reference envelope, not measured robot geometry.",
            "Triangle surface intersections do not classify closed-solid containment or semantically valid support.",
            "A finite support stencil and finite path samples cannot certify continuous contact or clearance.",
            "Pitch, roll, wheel contacts, suspension, localization uncertainty and dynamic obstacles are absent.",
            "Zone IDs refer to nominal source surfaces; RFID and hardware TF registration are not validated."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out),
        "pose_passes": {r["id"]: r["sampled_reference_pass"] for r in probes},
        "segment_passes": {r["id"]: r["sampled_reference_pass"] for r in paths},
        "hardware_reachability_certified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
