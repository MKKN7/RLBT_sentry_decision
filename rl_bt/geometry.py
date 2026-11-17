"""Conservative planar navigation proxy for the *simplified* alphaRM map."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def reference_config():
    path = ROOT / "data/references/alpharm_jax_review/upstream/jaxmarl/environments/rm/config.py"
    spec = importlib.util.spec_from_file_location("alpharm_static_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RmConfig()


def polygon_clearance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Signed distance to a possibly concave polygon; negative inside."""
    points = np.atleast_2d(points)
    inside = np.zeros(len(points), bool)
    distance = np.full(len(points), np.inf)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        t = np.clip(((points - a) @ edge) / max(edge @ edge, 1e-12), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points - a - t[:, None] * edge, axis=1))
        if abs(b[1] - a[1]) > 1e-12:
            crosses = ((a[1] > points[:, 1]) != (b[1] > points[:, 1]))
            x_cross = a[0] + (points[:, 1] - a[1]) * edge[0] / edge[1]
            inside ^= crosses & (points[:, 0] < x_cross)
    return np.where(inside, -distance, distance)


class PlanarProxy:
    """Disk clearance + eight-connected shortest paths; not Nav2/MPPI."""

    def __init__(self, resolution: float = 0.20):
        self.cfg = reference_config()
        self.resolution = resolution
        self.polygons = [np.asarray(p) for p in self.cfg.covers]
        x = np.arange(-self.cfg.bound_x, self.cfg.bound_x + 1e-6, resolution)
        y = np.arange(-self.cfg.bound_y, self.cfg.bound_y + 1e-6, resolution)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        all_xy = np.stack((xx, yy), axis=-1)
        free = self.clearance(all_xy.reshape(-1, 2)).reshape(xx.shape) >= (
            self.cfg.robot_radius + resolution / np.sqrt(2)
        )
        self.xy = all_xy[free]
        self.tree = cKDTree(self.xy)
        ids = np.full(free.shape, -1, dtype=int)
        ids[free] = np.arange(len(self.xy))
        edges, destinations, weights = [], [], []
        for i, j in np.argwhere(free):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = i + di, j + dj
                if not (0 <= a < len(x) and 0 <= b < len(y) and free[a, b]):
                    continue
                if di and dj and not (free[i, b] and free[a, j]):
                    continue
                edges.append(ids[i, j]); destinations.append(ids[a, b])
                weights.append(np.hypot(di, dj) * resolution)
        graph = csr_matrix((weights, (edges, destinations)), shape=(len(self.xy),) * 2)
        # Semantic locations are prototype choices derived from the upstream map.
        # They are not official UL/UC field annotations.
        home = np.mean(self.cfg.red_spawn, axis=0)
        lo, hi = np.min(self.cfg.red_l, axis=0), np.max(self.cfg.red_l, axis=0)
        r = self.cfg.center_radius * 0.65
        canonical = np.asarray([
            [0, 0], home, [0, r], [0, -r],
            [lo[0] - .6, lo[1] - .6], [hi[0] + .6, lo[1] - .6],
            [hi[0] + .6, hi[1] + .4], [-lo[0] + .6, -lo[1] + .6],
            [-hi[0] - .6, -lo[1] + .6], [-hi[0] - .6, -hi[1] - .4],
            [-2.4, -2.3], [2.4, 2.3], [0, 2.7], [0, -2.7], [2.5, 0],
        ])
        self.goal_ids = []
        for sign in (1, -1):
            self.goal_ids.append(self.tree.query(canonical * sign)[1])
        self.goal_ids = np.asarray(self.goal_ids)
        self.goals = self.xy[self.goal_ids]
        self.distances, self.predecessors = dijkstra(
            graph, directed=False, indices=self.goal_ids.reshape(-1), return_predecessors=True
        )
        self.distances = self.distances.reshape(2, 15, -1)
        self.predecessors = self.predecessors.reshape(2, 15, -1)

    def clearance(self, points):
        points = np.atleast_2d(points)
        return np.min([polygon_clearance(points, p) for p in self.polygons], axis=0)

    def node(self, position):
        return int(self.tree.query(position)[1])

    def costs(self, position, team):
        n = self.node(position)
        return self.distances[team, :, n] + np.linalg.norm(position - self.xy[n])

    def waypoint(self, position, team, goal):
        """goal is zero based here; callers own the task/candidate numbering."""
        n = self.node(position)
        if not np.isfinite(self.distances[team, goal, n]):
            return None
        path = [self.xy[n]]
        for _ in range(4):
            n = int(self.predecessors[team, goal, n])
            if n < 0:
                break
            path.append(self.xy[n])
        # Look ahead only when the segment keeps conservative disk clearance.
        for point in reversed(path):
            samples = np.linspace(position, point, 15)
            if np.min(self.clearance(samples)) >= self.cfg.robot_radius + .03:
                return point
        return path[0]
