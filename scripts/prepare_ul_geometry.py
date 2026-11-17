#!/usr/bin/env python3
"""Validate saved UL tessellation data and write research STL/semantic bounds.

No code in the downloaded community archive is executed. STEP-to-triangle
conversion provenance is preserved; this is not independent OCCT validation.
"""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    package = ROOT / "data/references/rmul_2027_community_map/unpacked/RMUL2027_map_transfer_20261016_STEP_R1"
    source = package / "src/guodong_sentry_sim/maps/rmul_2027"
    cad = ROOT / "assets/rmul_2027/source_cad/RMUL2027_3V3_V1.0.0.stp"
    geometry_file = source / "step_geometry.json.gz"
    geometry = json.loads(gzip.decompress(geometry_file.read_bytes()))
    measured = json.loads((source / "step_measurements.json").read_text())
    imported = json.loads((source / "step_import.json").read_text())
    cad_hash = sha(cad)
    if any(x["source_sha256"] != cad_hash for x in (geometry, measured, imported)):
        raise ValueError("Community geometry source differs from downloaded official STEP")
    if geometry["units"] != "metres":
        raise ValueError("Unexpected tessellation unit")
    faces = geometry["faces"]
    triangles = np.concatenate([np.asarray(f["triangles_m"], dtype=np.float64).reshape(-1, 3, 3)
                                 for f in faces])
    if not np.isfinite(triangles).all():
        raise ValueError("Non-finite geometry")
    actual_bounds = np.r_[triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))]
    np.testing.assert_allclose(actual_bounds, imported["bounds_m"], atol=1e-7)
    assert len(triangles) == imported["triangles"]
    verified = {}
    for name, entry in measured["measurements"].items():
        subset = [f for f in faces if f["occurrence"] == entry["occurrence"]
                  and f["face"] == entry["face_index"]]
        if len(subset) != 1:
            raise ValueError(f"Missing or duplicated measured B-rep face {name}")
        vertices = np.asarray(subset[0]["triangles_m"]).reshape(-1, 3)
        bounds = np.r_[vertices.min(axis=0), vertices.max(axis=0)]
        np.testing.assert_allclose(bounds, entry["bounds_m"], atol=1e-7)
        verified[name] = bounds.tolist()
    out = ROOT / "assets/rmul_2027/derived"
    out.mkdir(parents=True, exist_ok=True)
    stl = out / "field_reference.stl"
    if stl.exists():
        raise FileExistsError(stl)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(normal, axis=-1)
    normal /= np.maximum(norm[:, None], 1e-12)
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.zeros(len(triangles), dtype=dtype)
    records["normal"], records["vertices"] = normal, triangles
    stl.write_bytes(b"UL official STEP community tessellation; source hashes in manifest".ljust(80, b" ")
                    + struct.pack("<I", len(records)) + records.tobytes())
    zones = {name: {"bounds_xyz_minmax_m": verified[key], "source_face": measured["measurements"][key],
                    "purpose": "reference_surface_not_RFID_or_robot_body_zone_validation"}
             for name, key in [("red_supply", "red_supply_surface"),
                               ("red_highland_control", "red_highland_control"),
                               ("center_control", "center_control")]}
    for target, source_name in [("blue_supply", "red_supply"),
                                 ("blue_highland_control", "red_highland_control")]:
        b = np.asarray(zones[source_name]["bounds_xyz_minmax_m"])
        mirrored = [12-b[3], 8-b[4], b[2], 12-b[0], 8-b[1], b[5]]
        zones[target] = {"bounds_xyz_minmax_m": mirrored, "source": f"180_degree_mirror_of_{source_name}",
                         "purpose": "nominal_rule_drawing_symmetry_reference_requires_RFIDs_and_frame_registration"}
    manifest = {"artifact_type": "UL_reference_geometry_derived_from_saved_tessellation",
        "official_step": str(cad.relative_to(ROOT)), "official_step_sha256": cad_hash,
        "tessellation_source": str(geometry_file.relative_to(ROOT)), "tessellation_sha256": sha(geometry_file),
        "tessellation_conversion_independently_reproduced": False,
        "community_scripts_executed": False, "transform_description": geometry["transform"],
        "transform_applied_again": False, "bounds_m": actual_bounds.tolist(),
        "triangles": len(triangles), "measurement_face_bounds_verified": verified,
        "nominal_rule_size_xy_m": [12, 8], "geometry_file": str(stl.relative_to(ROOT)),
        "geometry_sha256": sha(stl), "navigation_and_hardware_registration_validated": False,
        "zones": zones,
        "limits": ["Community-generated tessellation; supplied STEP hash matches official download.",
            "Selected triangle face bounds checked against saved measurements, not full B-rep equivalence.",
            "Zone bounds are nominal visible surfaces, not validated RFID coverage or collision-free goals.",
            "Blue zones use official nominal field symmetry, not per-card survey data.",
            "No robot contact, slope connectivity, tunnel traversal or Nav2 validation implied."]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"triangles": len(triangles), "verified_bounds": len(verified),
                      "stl": str(stl.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
