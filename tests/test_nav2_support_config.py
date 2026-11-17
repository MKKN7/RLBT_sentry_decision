"""ROS-free regressions for strict observed-support costmap configuration.

The critical parameter scopes and combination rules were checked against the
vendored Nav2 1.1.20 sources in data/references/nav2_support_review. These tests
protect configuration safety invariants; they do not simulate the ROS plugins.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nav2_support_launcher", ROOT / "scripts/launch_nav2_probe.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


@pytest.fixture
def base_document():
    return yaml.safe_load((ROOT / "configs/nav2_probe.yaml").read_text())


def costmaps(document):
    return [document[name][name]["ros__parameters"]
            for name in ("global_costmap", "local_costmap")]


def test_observed_support_is_first_and_static_global_parameters_are_top_level(base_document):
    document = LAUNCHER.configure_terrain_prior(base_document, "isolated_probe", "observed")
    for params in costmaps(document):
        assert params["plugins"] == ["support_layer", "obstacle_layer", "inflation_layer"]
        layer = params["support_layer"]
        assert layer["plugin"] == "nav2_costmap_2d::StaticLayer"
        assert layer["enabled"] is True
        assert layer["map_topic"] == "/isolated_probe/terrain_support"
        assert layer["map_subscribe_transient_local"] is True
        assert layer["subscribe_to_updates"] is False
        # StaticLayer 1.1.20 reads these at costmap scope, not its own scope.
        assert params["use_maximum"] is False
        assert params["unknown_cost_value"] == 255
        assert params["lethal_cost_threshold"] == 100
        assert params["trinary_costmap"] is True
        assert params["track_unknown_space"] is True
        assert not {"use_maximum", "unknown_cost_value", "lethal_cost_threshold"} & layer.keys()


def test_observed_support_cannot_be_cleared_by_footprint_or_obstacle_rays(base_document):
    document = LAUNCHER.configure_terrain_prior(base_document, "isolated_probe", "observed")
    for params in costmaps(document):
        assert params["support_layer"]["footprint_clearing_enabled"] is False
        obstacle = params["obstacle_layer"]
        assert obstacle["footprint_clearing_enabled"] is False
        assert obstacle["terrain"]["marking"] is True
        assert obstacle["terrain"]["clearing"] is False
        # ObstacleLayer 1.1.20 mode 1 merges via updateWithMax. Mode 0 could
        # overwrite lethal support with an obstacle layer's measured-free cell.
        assert obstacle["combination_method"] == 1


def test_observed_footprint_contains_full_reference_envelope(base_document):
    document = LAUNCHER.configure_terrain_prior(base_document, "isolated_probe", "observed")
    for params in costmaps(document):
        vertices = np.asarray(json.loads(params["footprint"]), dtype=float)
        assert vertices.shape == (16, 2)
        assert np.isfinite(vertices).all()
        edges = np.roll(vertices, -1, axis=0) - vertices
        # Signed distance from origin to every CCW edge. Requiring all edges
        # outside .411 m rejects an inscribed polygon or a displaced footprint.
        inward_distances = (vertices[:, 0] * edges[:, 1]
                            - vertices[:, 1] * edges[:, 0]) / np.linalg.norm(edges, axis=1)
        assert np.all(inward_distances >= .411 - 1e-12)
        assert params["footprint_padding"] == 0.
        assert params["robot_radius"] >= .411
    critic = document["controller_server"]["ros__parameters"]["FollowPath"]["CostCritic"]
    assert critic["enabled"] is True
    assert critic["consider_footprint"] is True


def test_observed_global_and_local_resolution_match_support_grid(base_document):
    document = LAUNCHER.configure_terrain_prior(base_document, "isolated_probe", "observed")
    for params in costmaps(document):
        assert params["resolution"] == pytest.approx(.15)
    qualified = LAUNCHER.qualify_parameters(document, "isolated_probe")
    for name in ("global_costmap", "local_costmap"):
        assert qualified[f"/isolated_probe/{name}/{name}"]["ros__parameters"]["resolution"] == .15


def test_clear_mode_preserves_existing_plane_probe_configuration(base_document):
    expected = deepcopy(base_document)
    result = LAUNCHER.configure_terrain_prior(base_document, "isolated_probe", "clear")
    assert result is base_document
    assert result == expected
    for params in costmaps(result):
        assert params["resolution"] == .1
        assert params["track_unknown_space"] is False
        assert "support_layer" not in params["plugins"]
        assert params["obstacle_layer"]["terrain"]["clearing"] is True


def test_unknown_terrain_policy_is_rejected(base_document):
    with pytest.raises(ValueError, match="Unknown terrain prior"):
        LAUNCHER.configure_terrain_prior(base_document, "isolated_probe", "unsupported")
