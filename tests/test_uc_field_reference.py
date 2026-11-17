"""UC geometry provenance gates and prepared-geometry identity, without ROS."""
import hashlib
import json

import pytest

from rl_bt import uc_field_reference


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def prepared_field(tmp_path):
    source_bytes = b"temporary source mesh for provenance gate tests"
    source = tmp_path / "source.stl"
    source.write_bytes(source_bytes)
    field = tmp_path / "prepared"
    field.mkdir()
    prepared_bytes = b"temporary translated prepared mesh"
    (field / "field.stl").write_bytes(prepared_bytes)
    manifest = {
        "source": str(source),
        "source_sha256": sha256(source_bytes),
        "translation": [-.15, -.06, -.02],
        "triangles": 3,
        "collision_pieces": 2,
    }
    (field / "manifest.json").write_text(json.dumps(manifest))
    return field, source, manifest, prepared_bytes


def test_unreviewed_source_is_rejected_even_when_its_hash_matches(prepared_field):
    field, _, _, _ = prepared_field
    with pytest.raises(ValueError, match="Unreviewed UC source mesh"):
        uc_field_reference.prepared_field_reference(field)


def test_reviewed_claim_does_not_accept_changed_source_bytes(prepared_field, monkeypatch):
    field, source, manifest, _ = prepared_field
    monkeypatch.setitem(uc_field_reference.SOURCES, manifest["source_sha256"], "reviewed_test_source")
    source.write_bytes(b"changed after preparation")
    with pytest.raises(ValueError, match="source hash differs"):
        uc_field_reference.prepared_field_reference(field)


def test_audited_source_records_prepared_hash_and_unverified_match_semantics(prepared_field, monkeypatch):
    field, source, manifest, prepared_bytes = prepared_field
    monkeypatch.setitem(uc_field_reference.SOURCES, manifest["source_sha256"], "reviewed_test_source")
    result = uc_field_reference.prepared_field_reference(field)
    assert result["original_source"] == str(source)
    assert result["prepared_field"] == str(field.resolve())
    assert result["source_sha256"] == manifest["source_sha256"]
    assert result["prepared_mesh_sha256"] == sha256(prepared_bytes)
    assert result["source_manifest_sha256"] == sha256((field / "manifest.json").read_bytes())
    assert result["translation_already_applied"] == manifest["translation"]
    assert result["geometry_variant"] == "reviewed_test_source"
    assert result["contact_model_is_approximation"] is True
    assert result["RFID_regions_registered"] is False
    assert result["official_referee_coordinate_registration_verified"] is False


def test_navigation_identity_uses_prepared_geometry_and_changes_with_it(prepared_field, monkeypatch):
    field, _, manifest, prepared_bytes = prepared_field
    monkeypatch.setitem(uc_field_reference.SOURCES, manifest["source_sha256"], "reviewed_test_source")
    first_reference = uc_field_reference.prepared_field_reference(field)
    first = uc_field_reference.navigation_identity(first_reference)
    assert first == ("rmuc_2027_execution_validation", "reviewed_test_source",
                     "sha256:" + sha256(prepared_bytes))
    assert first[2] != "sha256:" + manifest["source_sha256"]
    (field / "field.stl").write_bytes(b"different prepared geometry from the same reviewed source")
    second_reference = uc_field_reference.prepared_field_reference(field)
    second = uc_field_reference.navigation_identity(second_reference)
    assert first_reference["source_sha256"] == second_reference["source_sha256"]
    assert first[:2] == second[:2]
    assert first[2] != second[2]
