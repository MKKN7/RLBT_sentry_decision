"""Conservative ground-only motion proxy for UL task-level courses.

This graph consumes the frozen reference STL and the user-selected simulation
robot envelope.  It is not Nav2/MPPI, an RFID map, a measured hardware clearance
certificate, or a replacement for live BT reachability validation.  Elevated
decks are deliberately unreachable in this ground-only course.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .layered_geometry import read_binary_stl
from .mesh_navigation import triangle_box_intersections


ROOT = Path(__file__).resolve().parents[1]


class ULNavigation:
    """8-connected, no-corner-cutting graph with cached goal distance trees.

    Each free graph cell represents a whole XY square cleared for the planning
    circle, including its extra margin.  Therefore connecting an arbitrary pose
    to its own cell centre cannot skip an intervening blocked cell.  Mesh
    winding is only used to select known upward near-ground support; collision
    queries are two-sided.  Unknown support is blocked.
    """

    def __init__(self, *, resolution_m=.1, triangles=None, envelope=None,
                 bounds_xy=((0., 0.), (12., 8.))):
        self.resolution_m = float(resolution_m)
        if not np.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("Positive finite resolution required")
        bounds = np.asarray(bounds_xy, dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError("Positive finite XY bounds required")
        self.origin_xy_m = bounds[0].copy()
        self.shape = tuple(np.ceil((bounds[1] - bounds[0]) / self.resolution_m - 1e-10).astype(int))
        if np.prod(self.shape) > 100_000:
            raise ValueError("Course grid exceeds bounded preprocessing size")
        self.bounds_xy_m = bounds
        self.envelope = (json.loads((ROOT / "configs/reference_robot_envelope.json").read_text())
                         if envelope is None else dict(envelope))
        self.radius_m = float(self.envelope["planning_radius_m"]) + float(self.envelope["extra_planar_margin_m"])
        self.height_m = float(self.envelope["planning_height_m"]) + float(self.envelope["extra_vertical_margin_m"])
        if not np.isfinite([self.radius_m, self.height_m]).all() or min(self.radius_m, self.height_m) <= 0:
            raise ValueError("Positive finite planning envelope required")
        if triangles is None:
            manifest = json.loads((ROOT / "assets/rmul_2027/derived/manifest.json").read_text())
            mesh_path = ROOT / manifest["geometry_file"]
            self.mesh_sha256 = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            if self.mesh_sha256 != manifest["geometry_sha256"]:
                raise ValueError("UL reference mesh differs from source manifest")
            triangles = read_binary_stl(mesh_path)
        else:
            self.mesh_sha256 = None
        triangles = np.asarray(triangles, dtype=float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles) or not np.isfinite(triangles).all():
            raise ValueError("Nonempty finite triangle mesh required")
        self.mesh_triangle_count = len(triangles)
        self.supported_cells, self.obstacle_cells = self._rasterize(triangles)
        unknown_or_obstacle = ~self.supported_cells | self.obstacle_cells
        # Padding makes the outside explicitly blocked. A raw occupied square
        # extends half a diagonal from its centre, and a destination square has
        # the same extent; using both diagonals is a conservative grid bound.
        padded = np.pad(~unknown_or_obstacle, 1, constant_values=False)
        clearance = distance_transform_edt(padded)[1:-1, 1:-1] * self.resolution_m
        self.geometric_free = clearance > self.radius_m + np.sqrt(2.) * self.resolution_m + 1e-10
        self.free_by_team = tuple(self._exclude_enemy_supply(team) for team in (0, 1))
        self._graphs = tuple(self._make_graph(free) for free in self.free_by_team)
        self._goal_trees = {}

    def _rasterize(self, triangles):
        """Ground ray support and exact slab/cell intersections, including walls."""
        support = np.zeros(self.shape, dtype=bool)
        obstacle = np.zeros(self.shape, dtype=bool)
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(normal, axis=1)
        lower, upper = triangles.min(axis=1), triangles.max(axis=1)
        # The ground support range explicitly excludes the -0.1 m floor bottom.
        support_faces = ((lower[:, 2] >= -1e-6) & (upper[:, 2] <= .015 + 1e-6)
                         & (normal[:, 2] > 0) & (normal[:, 2] >= np.cos(np.deg2rad(20.)) * norms))
        ground_top = .015
        slab_top = ground_top + self.height_m
        obstacle_faces = (upper[:, 2] > ground_top + 1e-6) & (lower[:, 2] <= slab_top + 1e-8)
        for face_id in np.flatnonzero(support_faces | obstacle_faces):
            tri = triangles[face_id]
            lo = np.maximum(np.floor((lower[face_id, :2] - self.origin_xy_m) / self.resolution_m - 1e-8).astype(int), 0)
            hi = np.minimum(np.floor((upper[face_id, :2] - self.origin_xy_m) / self.resolution_m + 1e-8).astype(int), np.asarray(self.shape) - 1)
            if np.any(lo > hi):
                continue
            xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
            cells = np.column_stack((xx.ravel(), yy.ravel()))
            xy = self.origin_xy_m + (cells + .5) * self.resolution_m
            if support_faces[face_id]:
                ab, ac = tri[1, :2] - tri[0, :2], tri[2, :2] - tri[0, :2]
                delta = xy - tri[0, :2]
                det = ab[0] * ac[1] - ab[1] * ac[0]
                u = (delta[:, 0] * ac[1] - delta[:, 1] * ac[0]) / det
                v = (ab[0] * delta[:, 1] - ab[1] * delta[:, 0]) / det
                hits = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1. + 1e-8)
                support[cells[hits, 0], cells[hits, 1]] = True
            if obstacle_faces[face_id]:
                centres = np.column_stack((xy, np.full(len(xy), (ground_top + slab_top) / 2)))
                local = tri[None, :, :] - centres[:, None, :]
                hits = triangle_box_intersections(local, [self.resolution_m / 2,
                    self.resolution_m / 2, (slab_top - ground_top) / 2])
                obstacle[cells[hits, 0], cells[hits, 1]] = True
        return support, obstacle

    @staticmethod
    def _team(team):
        if isinstance(team, bool) or team not in (0, 1):
            raise ValueError("team must be 0 (red) or 1 (blue)")
        return int(team)

    def _exclude_enemy_supply(self, team):
        # Bounds are nominal visible supply surfaces; course geometry only.
        lower, upper = (np.asarray([9.5, 0.]), np.asarray([12., 2.])) if team == 0 else (np.asarray([0., 6.]), np.asarray([2.5, 8.]))
        ix, iy = np.indices(self.shape)
        cell_min = self.origin_xy_m + np.stack((ix, iy), axis=-1) * self.resolution_m
        cell_max = cell_min + self.resolution_m
        separation = np.maximum(np.maximum(lower - cell_max, cell_min - upper), 0.)
        body_may_overlap = np.linalg.norm(separation, axis=-1) <= self.radius_m + 1e-10
        return self.geometric_free & ~body_may_overlap

    def _make_graph(self, free):
        rows, cols, costs = [], [], []
        nx, ny = self.shape
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            sx, sy = np.nonzero(free)
            tx, ty = sx + dx, sy + dy
            inside = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)
            sx, sy, tx, ty = sx[inside], sy[inside], tx[inside], ty[inside]
            valid = free[tx, ty]
            if dx and dy:
                valid &= free[sx, ty] & free[tx, sy]
            src, dst = sx[valid] * ny + sy[valid], tx[valid] * ny + ty[valid]
            weight = np.full(len(src), np.hypot(dx, dy) * self.resolution_m)
            rows.extend((src, dst)); cols.extend((dst, src)); costs.extend((weight, weight))
        n = nx * ny
        return csr_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    def _cell(self, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            return None
        cell = np.floor((xy - self.origin_xy_m) / self.resolution_m).astype(int)
        if np.any(cell < 0) or np.any(cell >= self.shape) or np.any(xy >= self.bounds_xy_m[1]):
            return None
        return tuple(cell)

    def is_free(self, xy, team):
        team = self._team(team)
        cell = self._cell(xy)
        return cell is not None and bool(self.free_by_team[team][cell])

    def _tree(self, goal_cell, team):
        goal_index = int(goal_cell[0] * self.shape[1] + goal_cell[1])
        key = (team, goal_index)
        if key not in self._goal_trees:
            distance, predecessor = dijkstra(self._graphs[team], directed=False,
                indices=goal_index, return_predecessors=True)
            # Bound caches when callers supply dynamic non-candidate goals.
            if len(self._goal_trees) >= 24:
                self._goal_trees.pop(next(iter(self._goal_trees)))
            self._goal_trees[key] = distance.astype(np.float32), predecessor
        return self._goal_trees[key]

    def plan(self, start_xy, goal_xy, team):
        """Return full Nx2 polyline including exact endpoints, or None."""
        team = self._team(team)
        if not self.is_free(start_xy, team) or not self.is_free(goal_xy, team):
            return None
        start, goal = self._cell(start_xy), self._cell(goal_xy)
        distances, predecessor = self._tree(goal, team)
        start_index, goal_index = start[0] * self.shape[1] + start[1], goal[0] * self.shape[1] + goal[1]
        if not np.isfinite(distances[start_index]):
            return None
        indices = [start_index]
        while indices[-1] != goal_index:
            nxt = int(predecessor[indices[-1]])
            if nxt < 0 or len(indices) > np.prod(self.shape):
                raise RuntimeError("Invalid cached navigation predecessor tree")
            indices.append(nxt)
        indices = np.asarray(indices)
        centres = self.origin_xy_m + (np.column_stack((indices // self.shape[1], indices % self.shape[1])) + .5) * self.resolution_m
        path = np.vstack((start_xy, centres, goal_xy)).astype(float)
        # Remove only exact adjacent duplicates, retaining graph corner checks.
        return path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-12]]

    def goal_candidates(self, team):
        """14 frozen course goals; 14=continue and 15=hold belong to caller."""
        team = self._team(team)
        anchors = [
            ("own_supply", [1.5, 7.], "resupply", "nominal_rule_surface_anchor"),
            ("center_control", [6., 4.], "capture", "nominal_rule_surface_anchor"),
            ("west_lane", [1.5, 4.], "position", "engineering_anchor"),
            ("east_lane", [10.5, 4.], "position", "engineering_anchor"),
            ("south_midfield", [6., 1.], "position", "engineering_anchor"),
            ("north_midfield", [6., 7.], "position", "engineering_anchor"),
            ("center_north_approach", [6., 5.9], "position", "engineering_anchor"),
            ("center_south_approach", [6., 2.1], "position", "engineering_anchor"),
            ("own_supply_exit", [1.5, 5.4], "position", "engineering_anchor"),
            ("enemy_supply_standoff", [10.5, 3.], "position", "engineering_anchor"),
            ("own_highland", [3.5, 4.8], "highland", "nominal_rule_surface_anchor"),
            ("enemy_highland", [8.5, 3.2], "highland", "nominal_rule_surface_anchor"),
            ("own_tunnel", [3.5, 7.46], "tunnel", "nominal_rule_surface_anchor"),
            ("enemy_tunnel", [8.5, .54], "tunnel", "nominal_rule_surface_anchor"),
        ]
        own_supply = np.asarray(anchors[0][1])
        if team:
            own_supply = np.asarray([12., 8.]) - own_supply
        result = []
        for identifier, (name, xy, task_type, source) in enumerate(anchors):
            xy = np.asarray(xy)
            if team:
                xy = np.asarray([12., 8.]) - xy
            # Ground-only course never promotes deck or tunnel probes to goals.
            legal = task_type not in ("highland", "tunnel") and self.plan(own_supply, xy, team) is not None
            result.append({"id": identifier, "name": name, "xy": xy.tolist(),
                "task_type": task_type, "static_legal": bool(legal), "source": source})
        return result

    def summary(self):
        return {"artifact_type": "UL_ground_only_course_motion_proxy",
            "resolution_m": self.resolution_m, "grid_shape_xy": [int(s) for s in self.shape],
            "planning_circle_radius_with_margin_m": self.radius_m,
            "planning_height_with_margin_m": self.height_m,
            "mesh_sha256": self.mesh_sha256, "mesh_triangle_count": self.mesh_triangle_count,
            "supported_cells": int(self.supported_cells.sum()),
            "obstacle_cells_before_inflation": int(self.obstacle_cells.sum()),
            "free_cells_red_blue": [int(f.sum()) for f in self.free_by_team],
            "actual_Nav2_MPPI_connected": False, "hardware_reachability_certified": False,
            "authoritative_live_BT_mask": False,
            "limitations": ["Ground-only upright circular envelope; no deck climbing.",
                "Mesh-derived nominal zones are not surveyed RFID or hardware frames.",
                "No dynamic contact, attitude, suspension, moving obstacles or localization errors.",
                "Conservative finite grid may reject physically possible routes."]}
