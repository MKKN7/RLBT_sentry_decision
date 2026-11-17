"""Vertical mesh samples preserving separate levels; not a navigation map.

Two-sided STL surfaces have no semantic ground/ceiling labels. These samples
describe intersections only: neither winding nor empty vertical space certifies
robot support, side clearance, legal zones or connectivity.
"""
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


def read_binary_stl(path):
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError("STL header is incomplete")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError("Expected complete binary STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                      ("attribute", "<u2")])
    triangles = np.frombuffer(data, dtype=dtype, offset=84)["vertices"].astype(np.float64)
    if not np.isfinite(triangles).all():
        raise ValueError("Mesh contains non-finite vertices")
    return triangles


def rasterize_vertical_surfaces(triangles, *, resolution=.1, merge_tolerance_m=1e-5,
                                max_slope_deg=20., bounds_xy=None):
    """Intersect XY cell-centre rays with projected triangles, without flattening.

    ``z_to_next_surface_m`` is the next strictly higher sampled intersection,
    not certified free clearance. No intersection means unknown, not free.
    Near-vertical faces are deliberately absent from the ray representation;
    full mesh collision/footprint checks must handle them separately.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("Expected nonempty (triangles,3,3) mesh")
    if not np.isfinite(triangles).all():
        raise ValueError("Mesh contains non-finite values")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Positive finite resolution required")
    if not np.isfinite(merge_tolerance_m) or merge_tolerance_m <= 0:
        raise ValueError("Positive finite merge tolerance required")
    if not 0 <= max_slope_deg < 90:
        raise ValueError("Slope threshold must be within [0,90)")
    if bounds_xy is None:
        bounds_xy = np.stack((triangles[..., :2].min(axis=(0, 1)),
                              triangles[..., :2].max(axis=(0, 1))))
    bounds = np.asarray(bounds_xy, dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Expected positive finite XY bounds")
    origin = np.floor(bounds[0] / resolution) * resolution
    shape = np.ceil((bounds[1] - origin) / resolution - 1e-10).astype(int)
    if np.prod(shape) > 5_000_000:
        raise ValueError("Requested grid exceeds bounded preprocessing size")
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    valid = (norm > 1e-12) & (np.abs(normal[:, 2]) > 1e-12)
    slope = np.full(len(triangles), 90.)
    slope[valid] = np.degrees(np.arccos(np.clip(np.abs(normal[valid, 2]) / norm[valid], 0, 1)))
    ids, heights, face_ids = [], [], []
    for face_id in np.flatnonzero(valid):
        tri = triangles[face_id]
        lower = np.ceil((tri[:, :2].min(axis=0) - origin) / resolution - .5 - 1e-9).astype(int)
        upper = np.floor((tri[:, :2].max(axis=0) - origin) / resolution - .5 + 1e-9).astype(int)
        lower = np.maximum(lower, 0); upper = np.minimum(upper, shape - 1)
        if np.any(lower > upper):
            continue
        xx, yy = np.meshgrid(np.arange(lower[0], upper[0] + 1),
                             np.arange(lower[1], upper[1] + 1), indexing="ij")
        cell = np.column_stack((xx.ravel(), yy.ravel()))
        xy = origin + (cell + .5) * resolution
        a, b, c = tri
        ab, ac = b[:2] - a[:2], c[:2] - a[:2]
        delta = xy - a[:2]
        determinant = ab[0] * ac[1] - ab[1] * ac[0]
        u = (delta[:, 0] * ac[1] - delta[:, 1] * ac[0]) / determinant
        v = (ab[0] * delta[:, 1] - ab[1] * delta[:, 0]) / determinant
        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        if not inside.any():
            continue
        ids.append(cell[inside, 0] * shape[1] + cell[inside, 1])
        heights.append(a[2] + u[inside] * (b[2] - a[2]) + v[inside] * (c[2] - a[2]))
        face_ids.append(np.full(int(inside.sum()), face_id, dtype=np.int32))
    if ids:
        column = np.concatenate(ids).astype(np.int32)
        z = np.concatenate(heights)
        faces = np.concatenate(face_ids)
        order = np.lexsort((faces, z, column))
        column, z, faces = column[order], z[order], faces[order]
        # Merge only near-coincident hits in one column, preserving every level.
        starts = np.r_[0, np.flatnonzero((column[1:] != column[:-1])
                         | (z[1:] - z[:-1] > merge_tolerance_m)) + 1]
        min_slope = np.minimum.reduceat(slope[faces], starts)
        column, z, faces = column[starts], z[starts], faces[starts]
    else:
        column = np.empty(0, np.int32); z = np.empty(0)
        faces = np.empty(0, np.int32); min_slope = np.empty(0)
    counts = np.bincount(column, minlength=int(np.prod(shape))).astype(np.int32)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    gap = np.full(len(z), np.nan)
    same_column = column[1:] == column[:-1]
    gap[np.flatnonzero(same_column)] = np.diff(z)[same_column]
    return {"origin_xy_m": origin, "resolution_m": np.asarray(resolution),
            "grid_shape_xy": shape.astype(np.int32), "column_offsets": offsets,
            "surface_z_m": z.astype(np.float32), "representative_face_id": faces,
            "min_incident_slope_deg": min_slope.astype(np.float32),
            "low_slope_surface_candidate": min_slope <= max_slope_deg + 1e-6,
            "z_to_next_surface_m": gap.astype(np.float32),
            "no_higher_sampled_surface": np.isnan(gap),
            "surface_count_xy": counts.reshape(tuple(shape)),
            "vertical_or_degenerate_face_count": np.asarray(int((~valid).sum()))}
