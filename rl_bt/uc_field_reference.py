"""Identity of the selected prepared UC geometry, separate from match semantics."""
import hashlib
import json
from pathlib import Path


DEFAULT_PREPARED_FIELD = "/tmp/rl_bt_nav_probe/fields/rmuc2027_original"
SOURCES = {
    "2835c0e755c255f757d810d260283c623fbd4a49a8fd986487c70ad2a89e9d10":
        "unmodified_rmuc_2027_source_with_approximate_contact_hulls",
    "97faa279776cc7b00a635dde51b1bc3dcf8ea91cb949fc73e88e28ea7b94905f":
        "previously_flattened_driveable_derivative",
}


def prepared_field_reference(directory):
    field = Path(directory).resolve(strict=True)
    manifest_path = field / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_hash = manifest["source_sha256"]
    if source_hash not in SOURCES:
        raise ValueError("Unreviewed UC source mesh; audit before executing")
    source = Path(manifest["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("UC source hash differs from prepared manifest")
    prepared_hash = hashlib.sha256((field / "field.stl").read_bytes()).hexdigest()
    return {"prepared_field": str(field), "original_source": str(source),
        "source_sha256": source_hash, "prepared_mesh_sha256": prepared_hash,
        "translation_already_applied": manifest["translation"],
        "triangles": manifest["triangles"], "collision_pieces": manifest["collision_pieces"],
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "geometry_variant": SOURCES[source_hash],
        "contact_model_is_approximation": True,
        "RFID_regions_registered": False,
        "official_referee_coordinate_registration_verified": False}


def navigation_identity(reference):
    return ("rmuc_2027_execution_validation", reference["geometry_variant"],
            "sha256:" + reference["prepared_mesh_sha256"])
