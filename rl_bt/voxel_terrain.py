"""Observed, gravity-aligned point-cloud terrain for the Nav2 adapter.

This module consumes estimated registered scans, never simulator triangles or
truth poses.  It retains vertical layers, fits local support planes and floods
only surfaces connected to the estimated chassis contact height.  Unobserved,
sparse and ambiguous multi-level cells stay unknown.  The resulting 2D grid is
a conservative local navigation input, not a hardware clearance certificate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VoxelTerrainConfig:
    """Initial engineering settings; only the 20 degree limit is user supplied."""

    resolution_m: float = .15
    z_resolution_m: float = .04
    radius_m: float = 6.
    max_age_s: float = 3.
    max_sample_age_s: float = .5
    max_voxels: int = 100_000
    max_slope_deg: float = 20.
    plane_neighborhood_m: float = .34
    min_plane_columns: int = 5
    plane_max_residual_m: float = .025
    max_layer_thickness_m: float = .065
    merge_vertical_gap_m: float = .065
    max_step_m: float = .025
    seed_radius_m: float = .9
    seed_height_tolerance_m: float = .045
    support_contact_offset_m: float = 0.
    body_height_with_margin_m: float = .255
    obstacle_above_support_m: float = .035
    self_exclusion_radius_m: float = .45

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"Finite {name} required")
        positive = (self.resolution_m, self.z_resolution_m, self.radius_m,
            self.max_age_s, self.max_sample_age_s, self.plane_neighborhood_m,
            self.plane_max_residual_m, self.max_layer_thickness_m,
            self.merge_vertical_gap_m, self.seed_radius_m,
            self.seed_height_tolerance_m, self.body_height_with_margin_m,
            self.obstacle_above_support_m)
        if min(positive) <= 0 or self.max_step_m < 0 or self.self_exclusion_radius_m < 0:
            raise ValueError("Positive sizes and nonnegative step/exclusion required")
        if not 0 <= self.max_slope_deg < 90:
            raise ValueError("Slope threshold must lie within [0,90)")
        if (isinstance(self.max_voxels, bool) or not isinstance(self.max_voxels, int)
                or self.max_voxels < 1 or isinstance(self.min_plane_columns, bool)
                or not isinstance(self.min_plane_columns, int) or self.min_plane_columns < 3):
            raise ValueError("Positive voxel limit and at least three plane columns required")
        if math.ceil(2 * self.radius_m / self.resolution_m) ** 2 > 200_000:
            raise ValueError("Terrain grid exceeds bounded runtime size")


@dataclass(frozen=True)
class TerrainGrid:
    frame_id: str
    sampled_at_s: float
    origin_xy_m: np.ndarray
    resolution_m: float
    occupancy_xy: np.ndarray
    support_height_xy_m: np.ndarray
    slope_xy_deg: np.ndarray
    obstacle_points_xyz: np.ndarray
    diagnostics: dict

    def ros_data(self):
        """ROS OccupancyGrid x-fastest row-major layout, with -1 unknown."""
        return self.occupancy_xy.T.ravel().tolist()


class VoxelTerrainMap:
    """Bounded rolling observations with explicit sample time/frame rejection.

    ``robot_position_xyz`` must be the estimated chassis reference pose in the
    same gravity-aligned frame as the scan.  Configure contact offset if that
    pose reference is not at the wheel contact plane.  No implicit TF or truth
    fallback is performed here.  Output cells are not footprint inflated;
    Nav2 must apply its independently configured robot envelope and inflation.
    """

    def __init__(self, config=None, *, frame_id="odom"):
        self.config = VoxelTerrainConfig() if config is None else config
        if not isinstance(self.config, VoxelTerrainConfig) or not frame_id:
            raise ValueError("VoxelTerrainConfig and nonempty fixed frame required")
        self.frame_id = str(frame_id)
        self._voxels = {}
        self._last_sample_s = None

    @staticmethod
    def _pose(robot_position_xyz):
        pose = np.asarray(robot_position_xyz, float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("Finite estimated robot XYZ required")
        return pose

    def update(self, points_xyz, *, sampled_at_s, now_s, robot_position_xyz, frame_id):
        """Accept one scan; return rejection reason or ``accepted``.

        Duplicate/out-of-order, future and stale scans do not mutate the map or
        refresh its watchdog.  Each voxel stores the most recent scan centroid,
        rather than indefinitely averaging moving objects into the terrain.
        """
        pose = self._pose(robot_position_xyz)
        if frame_id != self.frame_id:
            return "frame_mismatch"
        if not np.isfinite([sampled_at_s, now_s]).all():
            return "nonfinite_timestamp"
        if sampled_at_s > now_s + 1e-6:
            return "future_sample"
        if now_s - sampled_at_s > self.config.max_sample_age_s:
            return "stale_sample"
        if self._last_sample_s is not None and sampled_at_s <= self._last_sample_s:
            return "out_of_order"
        points = np.asarray(points_xyz, float)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("Expected (N,3) estimated registered scan")
        points = points[np.isfinite(points).all(axis=1)]
        cfg = self.config
        distance = np.linalg.norm(points[:, :2] - pose[:2], axis=1)
        points = points[(distance <= cfg.radius_m)
                        & (distance >= cfg.self_exclusion_radius_m)
                        & (np.abs(points[:, 2] - pose[2]) <= 3.)]
        self._prune(now_s, pose)
        if len(points):
            scale = np.asarray([cfg.resolution_m, cfg.resolution_m, cfg.z_resolution_m])
            keys, inverse = np.unique(np.floor(points / scale).astype(np.int64),
                                      axis=0, return_inverse=True)
            count = np.bincount(inverse)
            centres = np.column_stack([np.bincount(inverse, weights=points[:, i]) / count
                                       for i in range(3)])
            for key, centre in zip(keys, centres):
                self._voxels[tuple(key)] = (centre, float(sampled_at_s))
            if len(self._voxels) > cfg.max_voxels:
                newest = sorted(self._voxels, key=lambda k: self._voxels[k][1], reverse=True)
                self._voxels = {k: self._voxels[k] for k in newest[:cfg.max_voxels]}
        self._last_sample_s = float(sampled_at_s)
        return "accepted"

    def _prune(self, now_s, pose):
        cfg = self.config
        self._voxels = {key: value for key, value in self._voxels.items()
            if now_s - value[1] <= cfg.max_age_s
            and np.linalg.norm(value[0][:2] - pose[:2]) <= cfg.radius_m}

    def snapshot(self, *, robot_position_xyz, now_s):
        pose = self._pose(robot_position_xyz)
        if not math.isfinite(now_s):
            raise ValueError("Finite current time required")
        cfg = self.config
        self._prune(now_s, pose)
        origin_cell = np.floor((pose[:2] - cfg.radius_m) / cfg.resolution_m).astype(int)
        origin = origin_cell * cfg.resolution_m
        side = int(math.ceil(2 * cfg.radius_m / cfg.resolution_m)) + 1
        shape = (side, side)
        occupancy = np.full(shape, -1, np.int8)
        height = np.full(shape, np.nan, np.float32)
        slope_grid = np.full(shape, np.nan, np.float32)
        columns = {}
        for key, (point, _) in self._voxels.items():
            cell = (int(key[0]), int(key[1]))
            columns.setdefault(cell, []).append(point)
        layers, by_column = [], {}
        # A column may contain ground, ceiling and upper deck independently.
        for cell, column_points in columns.items():
            points = np.asarray(column_points)
            points = points[np.argsort(points[:, 2])]
            cuts = np.r_[0, np.flatnonzero(np.diff(points[:, 2]) > cfg.merge_vertical_gap_m) + 1, len(points)]
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                group = points[lo:hi]
                index = len(layers)
                layers.append({"cell": cell, "point": group.mean(axis=0),
                    "min_z": float(group[:, 2].min()), "max_z": float(group[:, 2].max()),
                    "walkable": False, "slope": np.nan, "plane": None})
                by_column.setdefault(cell, []).append(index)
        plane_radius_cells = int(math.ceil(cfg.plane_neighborhood_m / cfg.resolution_m))
        for layer in layers:
            if layer["max_z"] - layer["min_z"] > cfg.max_layer_thickness_m:
                continue
            cell, point = layer["cell"], layer["point"]
            neighbors = []
            for dx in range(-plane_radius_cells, plane_radius_cells + 1):
                for dy in range(-plane_radius_cells, plane_radius_cells + 1):
                    ids = by_column.get((cell[0] + dx, cell[1] + dy), ())
                    if not ids:
                        continue
                    near = min(ids, key=lambda i: abs(layers[i]["point"][2] - point[2]))
                    other = layers[near]
                    separation = np.linalg.norm(other["point"][:2] - point[:2])
                    if (separation <= cfg.plane_neighborhood_m + 1e-9
                            and abs(other["point"][2] - point[2]) <= .25
                            and other["max_z"] - other["min_z"] <= cfg.max_layer_thickness_m):
                        neighbors.append(other["point"])
            if len(neighbors) < cfg.min_plane_columns:
                continue
            neighbors = np.asarray(neighbors)
            # Centre coordinates to keep fitting numerically stable in large map frames.
            design = np.column_stack((neighbors[:, :2] - point[:2], np.ones(len(neighbors))))
            if np.linalg.matrix_rank(design) < 3:
                continue
            coefficients, *_ = np.linalg.lstsq(design, neighbors[:, 2], rcond=None)
            residual = float(np.abs(design @ coefficients - neighbors[:, 2]).max())
            angle = float(np.degrees(np.arctan(np.linalg.norm(coefficients[:2]))))
            layer["slope"] = angle
            layer["plane"] = coefficients
            layer["walkable"] = (angle <= cfg.max_slope_deg + 1e-6
                                  and residual <= cfg.plane_max_residual_m + 1e-9)
        # Flood from observed, near-contact-height support around the robot.
        # An unrelated low-slope roof cannot become ground merely for being flat.
        contact_z = pose[2] - cfg.support_contact_offset_m
        seed_candidates = [i for i, layer in enumerate(layers) if layer["walkable"]
            and np.linalg.norm(layer["point"][:2] - pose[:2]) <= cfg.seed_radius_m
            and abs(layer["point"][2] - contact_z) <= cfg.seed_height_tolerance_m]
        # Start from one height-consistent connected component. Seeding every
        # near-contact surface would teleport across a low isolated platform
        # when pose-height uncertainty exceeds the permitted physical step.
        seeds = [min(seed_candidates, key=lambda i: (
            abs(layers[i]["point"][2] - contact_z),
            np.linalg.norm(layers[i]["point"][:2] - pose[:2])))] if seed_candidates else []
        reachable = set(seeds)
        queue = deque(seeds)
        while queue:
            index = queue.popleft()
            source = layers[index]
            x, y = source["cell"]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for other_id in by_column.get((x + dx, y + dy), ()):
                    if other_id in reachable:
                        continue
                    other = layers[other_id]
                    if not other["walkable"]:
                        continue
                    delta_xy = other["point"][:2] - source["point"][:2]
                    # Evaluate both local planes across the border: a flat
                    # raised step must not inherit the permitted ramp gradient.
                    predicted_delta = [float(source["plane"][:2] @ delta_xy),
                                       float(other["plane"][:2] @ delta_xy)]
                    actual_delta = float(other["point"][2] - source["point"][2])
                    if max(abs(actual_delta - p) for p in predicted_delta) <= cfg.max_step_m + 1e-9:
                        reachable.add(other_id)
                        queue.append(other_id)
        obstacles = []
        ambiguous = 0
        for cell, ids in by_column.items():
            ix, iy = np.asarray(cell) - origin_cell
            if not (0 <= ix < side and 0 <= iy < side):
                continue
            supported = [layers[i] for i in ids if i in reachable]
            support = min(supported, key=lambda x: abs(x["point"][2] - contact_z)) if supported else None
            if len(supported) > 1:
                # Nav2's 2D grid cannot encode two simultaneous navigable levels.
                ambiguous += 1
                continue
            if support is None:
                low_z, high_z = contact_z + cfg.obstacle_above_support_m, contact_z + cfg.body_height_with_margin_m
            else:
                low_z = support["point"][2] + cfg.obstacle_above_support_m
                high_z = support["point"][2] + cfg.body_height_with_margin_m
            hits = [point for point in columns[cell] if low_z - 1e-9 <= point[2] <= high_z + 1e-9]
            if hits:
                occupancy[ix, iy] = 100
                obstacles.extend(hits)
            elif support is not None:
                occupancy[ix, iy] = 0
                height[ix, iy] = support["point"][2]
                slope_grid[ix, iy] = support["slope"]
        diagnostics = {"schema": "observed_voxel_terrain_v1", "config": asdict(cfg),
            "observed_voxels": len(self._voxels), "observed_columns": len(columns),
            "separate_surface_layers": len(layers), "support_seeds": len(seeds),
            "support_seed_candidates": len(seed_candidates),
            "reachable_support_layers": len(reachable), "ambiguous_multi_level_columns": ambiguous,
            "free_cells": int((occupancy == 0).sum()), "occupied_cells": int((occupancy == 100).sum()),
            "unknown_cells": int((occupancy == -1).sum()),
            "last_scan_at_s": self._last_sample_s,
            "source": "estimated_registered_scan_only", "simulator_truth_used": False,
            "unknown_is_free": False, "footprint_inflated": False,
            "hardware_clearance_certified": False}
        return TerrainGrid(self.frame_id, float(now_s), origin, cfg.resolution_m,
            occupancy, height, slope_grid,
            np.asarray(obstacles, dtype=np.float32).reshape((-1, 3)), diagnostics)
