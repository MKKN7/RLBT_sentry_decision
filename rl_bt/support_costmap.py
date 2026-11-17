"""Conservative 2D projections of observed support, without truth geometry.

The semantic grid retains unknown support.  Its strict navigation projection
turns unknown into occupied so a ROS StaticLayer can block it independently of
costmap unknown-space options.  Grid cells are uninflated; the separate path
check covers a circular robot envelope over entire path segments.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from rl_bt.voxel_terrain import TerrainGrid


@dataclass(frozen=True)
class FixedStartSupportPrior:
    """Explicit engineering assumption at one immutable session start.

    This is not a scan observation or a moving-footprint clearing operation.
    Cells intersecting the fixed disk may be assumed supported, including their
    full raster cells.  Observed occupied cells always take precedence.
    """

    center_xy_m: tuple[float, float]
    radius_m: float = .411

    def __post_init__(self):
        center = np.asarray(self.center_xy_m, float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Finite fixed initial XY required")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("Positive finite initial support radius required")
        object.__setattr__(self, "center_xy_m", tuple(float(v) for v in center))


@dataclass(frozen=True)
class SupportCostmap:
    frame_id: str
    sampled_at_s: float
    origin_xy_m: np.ndarray
    resolution_m: float
    occupancy_xy: np.ndarray
    strict_occupancy_xy: np.ndarray
    report: dict

    def ros_data(self, *, strict=False):
        """OccupancyGrid x-fastest order; select strict explicitly."""
        values = self.strict_occupancy_xy if strict else self.occupancy_xy
        return values.T.ravel().tolist()


@dataclass(frozen=True)
class SupportCostmapProjector:
    """Session projection policy: configure a fixed start prior only once."""

    initial_support_prior: FixedStartSupportPrior | None = None

    def __post_init__(self):
        if self.initial_support_prior is not None and not isinstance(
                self.initial_support_prior, FixedStartSupportPrior):
            raise TypeError("Explicit FixedStartSupportPrior or None required")

    def project(self, grid: TerrainGrid):
        return project_support_costmap(grid,
            initial_support_prior=self.initial_support_prior)


def _grid_values(grid):
    if not isinstance(grid, (TerrainGrid, SupportCostmap)):
        raise TypeError("TerrainGrid or SupportCostmap required")
    origin = np.asarray(grid.origin_xy_m, float)
    occupancy = np.asarray(grid.occupancy_xy)
    resolution = float(grid.resolution_m)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("Finite grid origin XY required")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Positive finite grid resolution required")
    if occupancy.ndim != 2 or min(occupancy.shape) < 1:
        raise ValueError("Nonempty 2D occupancy required")
    if not np.isin(occupancy, (-1, 0, 100)).all():
        raise ValueError("Occupancy must contain only -1, 0 and 100")
    return origin, resolution, occupancy


def project_support_costmap(grid: TerrainGrid, *, initial_support_prior=None):
    """Copy observed labels, optionally apply one fixed disk, and block unknown.

    The caller must reuse the same ``FixedStartSupportPrior`` for all rolling
    snapshots.  No current robot pose parameter is accepted.  No unknown cells
    beyond the explicitly reported fixed initial disk are filled.
    """
    origin, resolution, source = _grid_values(grid)
    if not isinstance(grid, TerrainGrid):
        raise TypeError("Projection input must be TerrainGrid")
    if initial_support_prior is not None and not isinstance(
            initial_support_prior, FixedStartSupportPrior):
        raise TypeError("Explicit FixedStartSupportPrior or None required")
    semantic = source.astype(np.int8, copy=True)
    filled = 0
    intersected = 0
    retained_occupied = 0
    prior_report = None
    if initial_support_prior is not None:
        prior = initial_support_prior
        lower = origin + np.indices(source.shape).transpose(1, 2, 0) * resolution
        center = np.asarray(prior.center_xy_m)
        delta = np.maximum(np.maximum(lower - center, center - (lower + resolution)), 0.)
        in_disk = np.sum(delta ** 2, axis=-1) <= prior.radius_m ** 2 + 1e-12
        to_fill = in_disk & (semantic == -1)
        filled = int(to_fill.sum())
        intersected = int(in_disk.sum())
        retained_occupied = int((in_disk & (semantic == 100)).sum())
        semantic[to_fill] = 0
        prior_report = {
            "center_xy_m": list(prior.center_xy_m), "radius_m": prior.radius_m,
            "fixed_in_session_frame": True, "intersected_cells": intersected,
            "unknown_cells_assumed_supported": filled,
            "occupied_cells_preserved": retained_occupied,
            "geometry_assumption": "Full raster cells intersecting the fixed initial disk are supported; occupied observations override the assumption.",
            "scan_observation": False,
        }
    strict = np.where(semantic == -1, 100, semantic).astype(np.int8)
    report = {
        "simulator_truth_used": False, "mesh_geometry_used": False,
        "unknown_is_free": False, "strict_unknown_value": 100,
        "source_free_cells": int((source == 0).sum()),
        "source_occupied_cells": int((source == 100).sum()),
        "source_unknown_cells": int((source == -1).sum()),
        "projected_free_cells": int((semantic == 0).sum()),
        "projected_occupied_cells": int((semantic == 100).sum()),
        "projected_unknown_cells": int((semantic == -1).sum()),
        "initial_support_prior": prior_report,
        "footprint_inflated": False,
    }
    return SupportCostmap(grid.frame_id, grid.sampled_at_s, origin.copy(),
                          resolution, semantic, strict, report)


def _first_disk_cell_contact(a, b, lower, upper, radius):
    """First t where the segment's disk intersects the closed cell square.

    Distance to a square is piecewise quadratic along a segment.  Splitting at
    its four edge crossings avoids center-only sampling and catches tiny holes
    even between sparse path waypoints.
    """
    delta = b - a
    breaks = [0., 1.]
    for axis in range(2):
        if abs(delta[axis]) > 1e-15:
            for boundary in (lower[axis], upper[axis]):
                t = (boundary - a[axis]) / delta[axis]
                if 0. < t < 1.:
                    breaks.append(float(t))
    breaks = sorted(set(breaks))
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        mid = a + ((lo + hi) / 2.) * delta
        alpha, beta = np.zeros(2), np.zeros(2)
        for axis in range(2):
            if mid[axis] < lower[axis]:
                alpha[axis], beta[axis] = -delta[axis], lower[axis] - a[axis]
            elif mid[axis] > upper[axis]:
                alpha[axis], beta[axis] = delta[axis], a[axis] - upper[axis]
        aa = float(alpha @ alpha)
        bb = float(2. * alpha @ beta)
        cc = float(beta @ beta - radius ** 2)
        if (aa * lo + bb) * lo + cc <= 1e-12:
            return lo
        if aa <= 1e-24:
            continue
        discriminant = bb * bb - 4. * aa * cc
        if discriminant < -1e-12:
            continue
        root = (-bb - math.sqrt(max(0., discriminant))) / (2. * aa)
        if lo - 1e-12 <= root <= hi + 1e-12:
            return min(hi, max(lo, root))
    return None


def path_support_check(grid, path_xy, *, radius_m=.411):
    """Check the continuous circular footprint against all intersected cells.

    Unknown, occupied and outside-map footprint areas reject the path.  Frame,
    map freshness and intended-action identity remain external checks.  The
    reported failure position is the earliest disk contact on the first failing
    segment, not just the first blocked cell center or next path waypoint.
    """
    origin, resolution, occupancy = _grid_values(grid)
    path = np.asarray(path_xy, float)
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 1 or not np.isfinite(path).all():
        raise ValueError("Nonempty finite (N,2) path required")
    if not math.isfinite(radius_m) or radius_m < 0:
        raise ValueError("Finite nonnegative footprint radius required")
    safe_lower = origin + radius_m
    safe_upper = origin + np.asarray(occupancy.shape) * resolution - radius_m
    checked = set()
    segments = zip(path[:-1], path[1:]) if len(path) > 1 else [(path[0], path[0])]
    for segment_index, (a, b) in enumerate(segments):
        failures = []
        if np.any(a < safe_lower - 1e-12) or np.any(a > safe_upper + 1e-12):
            return {"accepted": False, "reason": "outside_map",
                "failure_segment_index": segment_index,
                "failure_segment_fraction": 0.,
                "failure_position_xy_m": a.tolist(), "failure_cell_xy": None,
                "radius_m": radius_m, "checked_nonfree_cells": len(checked)}
        delta = b - a
        checked_fraction = 1.
        if np.any(b < safe_lower - 1e-12) or np.any(b > safe_upper + 1e-12):
            exit_times = []
            for axis in range(2):
                if b[axis] < safe_lower[axis]:
                    exit_times.append((safe_lower[axis] - a[axis]) / delta[axis])
                elif b[axis] > safe_upper[axis]:
                    exit_times.append((safe_upper[axis] - a[axis]) / delta[axis])
            checked_fraction = float(min(exit_times))
            failures.append((checked_fraction, "outside_map", None))
        # Clip the geometry calculation at the first map exit.  This bounds
        # numerical scale even for a finite but extremely distant endpoint.
        checked_b = a + checked_fraction * delta
        # Restrict work to the actual grid; out-of-map sweeps are rejected above.
        with np.errstate(over="ignore"):
            low_float = (np.minimum(a, checked_b) - radius_m - origin) / resolution
            high_float = (np.maximum(a, checked_b) + radius_m - origin) / resolution
        bounds = np.asarray(occupancy.shape)
        low = np.floor(np.nextafter(np.clip(low_float, -1., bounds), -np.inf)).astype(np.int64)
        high = np.floor(np.clip(high_float, -1., bounds)).astype(np.int64)
        low = np.maximum(low, 0)
        high = np.minimum(high, np.asarray(occupancy.shape) - 1)
        for ix in range(int(low[0]), int(high[0]) + 1):
            for iy in range(int(low[1]), int(high[1]) + 1):
                cell = (ix, iy)
                if occupancy[cell] == 0:
                    continue
                cell_lower = origin + np.asarray(cell) * resolution
                contact = _first_disk_cell_contact(a, checked_b, cell_lower,
                    cell_lower + resolution, radius_m)
                if contact is not None:
                    checked.add(cell)
                    failures.append((contact * checked_fraction, "unknown_support" if occupancy[cell] == -1
                                     else "occupied", cell))
        if failures:
            contact, reason, cell = min(failures, key=lambda item: item[0])
            return {"accepted": False, "reason": reason,
                "failure_segment_index": segment_index,
                "failure_segment_fraction": contact,
                "failure_position_xy_m": (a + contact * delta).tolist(),
                "failure_cell_xy": list(cell) if cell is not None else None,
                "radius_m": radius_m, "checked_nonfree_cells": len(checked)}
    return {"accepted": True, "reason": "supported", "radius_m": radius_m,
        "failure_segment_index": None, "failure_segment_fraction": None,
        "failure_position_xy_m": None, "failure_cell_xy": None,
        "checked_nonfree_cells": len(checked)}
