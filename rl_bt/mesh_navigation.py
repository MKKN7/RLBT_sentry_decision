"""Offline, mesh-surface checks for explicit navigation reference envelopes.

This is deliberately separate from Nav2.  A box is an upright reference
envelope, not the wheel/contact model of the real robot.  Surface intersections
and a finite support stencil cannot certify closed-solid containment, suspension
motion or continuously collision-free execution.  Callers must not turn a pass
into an authoritative hardware reachability mask.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ReferenceEnvelope:
    """User-supplied box dimensions; no unverified robot defaults are provided."""

    length_m: float
    width_m: float
    body_height_m: float
    belly_clearance_m: float

    def __post_init__(self):
        dimensions = (self.length_m, self.width_m, self.body_height_m,
                      self.belly_clearance_m)
        if not all(math.isfinite(v) for v in dimensions):
            raise ValueError("Envelope dimensions must be finite")
        if min(dimensions[:3]) <= 0 or self.belly_clearance_m <= 0:
            raise ValueError("Positive dimensions and belly clearance required")


def _rotation(yaw_rad):
    if not math.isfinite(yaw_rad):
        raise ValueError("Finite yaw required")
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def triangle_box_intersections(triangles_local, half_extents, *, tolerance_m=1e-8):
    """Triangle/closed-box overlap via the 13 separating-axis tests.

    The box is centred at zero.  Touching is conservatively an intersection;
    winding is irrelevant and vertical side walls participate in the test.
    """
    tri = np.asarray(triangles_local, dtype=np.float64)
    half = np.asarray(half_extents, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or not np.isfinite(tri).all():
        raise ValueError("Finite (N,3,3) triangles required")
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError("Positive finite box half extents required")
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Nonnegative finite tolerance required")
    if not len(tri):
        return np.zeros(0, dtype=bool)
    edges = np.roll(tri, -1, axis=1) - tri
    axes = [np.broadcast_to(np.eye(3)[None, :, :], (len(tri), 3, 3)),
            np.cross(edges[:, 0], edges[:, 1])[:, None, :]]
    axes.extend(np.cross(edges, axis) for axis in np.eye(3))
    axes = np.concatenate(axes, axis=1)
    norms = np.linalg.norm(axes, axis=-1)
    axes = axes / np.maximum(norms[..., None], 1e-30)
    projection = np.einsum("nvc,nac->nva", tri, axes)
    radius = np.einsum("nac,c->na", np.abs(axes), half)
    separated = ((projection.min(axis=1) > radius + tolerance_m)
                 | (projection.max(axis=1) < -radius - tolerance_m))
    # Degenerate cross-product axes provide no separating information.
    separated &= norms > 1e-12
    return ~separated.any(axis=1)


class MeshSurfaceQuery:
    """Broad-phase AABBs and exact triangle/OBB surface intersection queries."""

    def __init__(self, triangles):
        triangles = np.asarray(triangles, dtype=np.float64)
        if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
                or not len(triangles) or not np.isfinite(triangles).all()):
            raise ValueError("Nonempty finite (N,3,3) mesh required")
        self.triangles = triangles.copy()
        self.lower = triangles.min(axis=1)
        self.upper = triangles.max(axis=1)
        normal = np.cross(triangles[:, 1] - triangles[:, 0],
                          triangles[:, 2] - triangles[:, 0])
        norm = np.linalg.norm(normal, axis=1)
        self.vertical_or_degenerate = (norm <= 1e-12) | (np.abs(normal[:, 2]) <= 1e-12)
        self.slopes_deg = np.degrees(np.arccos(np.clip(
            np.abs(normal[:, 2]) / np.maximum(norm, 1e-30), 0., 1.)))

    def intersecting_faces(self, *, center_xyz_m, half_extents_m, yaw_rad=0.):
        center, half = np.asarray(center_xyz_m, float), np.asarray(half_extents_m, float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Finite XYZ centre required")
        if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
            raise ValueError("Positive finite XYZ half extents required")
        rotation = _rotation(yaw_rad)
        world_half = np.abs(rotation) @ half
        candidates = np.flatnonzero(np.all(self.upper >= center - world_half - 1e-8, axis=1)
                                    & np.all(self.lower <= center + world_half + 1e-8, axis=1))
        local = (self.triangles[candidates] - center) @ rotation
        return candidates[triangle_box_intersections(local, half)]

    def vertical_hits(self, xy_m):
        """Return distinct (height, minimum incident slope) samples at one XY.

        Neither winding nor height assigns semantic ground versus ceiling.
        """
        xy = np.asarray(xy_m, float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Finite XY sample required")
        candidates = np.flatnonzero(~self.vertical_or_degenerate
            & np.all(self.lower[:, :2] <= xy + 1e-9, axis=1)
            & np.all(self.upper[:, :2] >= xy - 1e-9, axis=1))
        if not len(candidates):
            return np.empty((0, 2))
        tri = self.triangles[candidates]
        a, ab, ac = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        delta = xy - a[:, :2]
        det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        u = (delta[:, 0] * ac[:, 1] - delta[:, 1] * ac[:, 0]) / det
        v = (ab[:, 0] * delta[:, 1] - ab[:, 1] * delta[:, 0]) / det
        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        z = a[inside, 2] + u[inside] * ab[inside, 2] + v[inside] * ac[inside, 2]
        slope = self.slopes_deg[candidates[inside]]
        if not len(z):
            return np.empty((0, 2))
        order = np.argsort(z)
        z, slope = z[order], slope[order]
        starts = np.r_[0, np.flatnonzero(np.diff(z) > 1e-6) + 1]
        return np.column_stack((z[starts], np.minimum.reduceat(slope, starts)))

    def probe_upright_pose(self, *, xy_m, support_z_m, yaw_rad,
                           envelope: ReferenceEnvelope, support_tolerance_m,
                           max_slope_deg=20., support_stencil=3):
        """Check an explicit support height and an upright box at a fixed pose.

        The full footprint receives a regular support stencil including corners.
        A support hit must be within the caller's tolerance of the supplied
        support height; no automatic jump to an overhead or underground layer.
        """
        xy = np.asarray(xy_m, float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(support_z_m):
            raise ValueError("Finite XY and explicit support height required")
        if not math.isfinite(support_tolerance_m) or support_tolerance_m < 0:
            raise ValueError("Nonnegative finite support tolerance required")
        if not math.isfinite(max_slope_deg) or not 0 <= max_slope_deg < 90:
            raise ValueError("Slope threshold must be within [0,90)")
        if isinstance(support_stencil, bool) or not isinstance(support_stencil, int) or support_stencil < 2:
            raise ValueError("Support stencil must have at least two samples per axis")
        rotation = _rotation(yaw_rad)
        sx = np.linspace(-envelope.length_m / 2, envelope.length_m / 2, support_stencil)
        sy = np.linspace(-envelope.width_m / 2, envelope.width_m / 2, support_stencil)
        points = np.asarray([(x, y, 0.) for x in sx for y in sy]) @ rotation.T
        points[:, :2] += xy
        supported, samples = [], []
        for point in points:
            hits = self.vertical_hits(point[:2])
            eligible = hits[(np.abs(hits[:, 0] - support_z_m) <= support_tolerance_m + 1e-8)
                            & (hits[:, 1] <= max_slope_deg + 1e-6)]
            supported.append(bool(len(eligible)))
            samples.append({"xy_m": point[:2].tolist(),
                "eligible_surface_z_m": eligible[:, 0].tolist(),
                "supported": supported[-1]})
        half = np.asarray([envelope.length_m, envelope.width_m, envelope.body_height_m]) / 2
        center = np.r_[xy, support_z_m + envelope.belly_clearance_m + half[2]]
        faces = self.intersecting_faces(center_xyz_m=center, half_extents_m=half, yaw_rad=yaw_rad)
        reasons = []
        if not all(supported):
            reasons.append("missing_or_wrong_layer_support_sample")
        if len(faces):
            reasons.append("body_box_intersects_mesh_surface")
        return {"sampled_reference_pass": not reasons, "reasons": reasons,
            "xy_m": xy.tolist(), "support_z_m": float(support_z_m), "yaw_rad": float(yaw_rad),
            "envelope": asdict(envelope), "support_tolerance_m": float(support_tolerance_m),
            "max_slope_deg": float(max_slope_deg), "support_samples": samples,
            "intersecting_face_ids": faces.tolist(), "hardware_reachability_certified": False}

    def sweep_upright_segment(self, *, start_xyzh, end_xyzh, envelope,
                              support_tolerance_m, max_translation_step_m,
                              max_yaw_step_rad, max_slope_deg=20.):
        """Sample a segment including endpoints; no continuous sweep claim.

        Pose order is x, y, explicit support z, yaw.  Angular interpolation takes
        the shortest wrapped path.  Both sampling resolutions are explicit.
        """
        start, end = np.asarray(start_xyzh, float), np.asarray(end_xyzh, float)
        if (start.shape != (4,) or end.shape != (4,) or not np.isfinite(start).all()
                or not np.isfinite(end).all()):
            raise ValueError("Finite start and end XY/support-Z/yaw required")
        if (not math.isfinite(max_translation_step_m) or max_translation_step_m <= 0
                or not math.isfinite(max_yaw_step_rad) or max_yaw_step_rad <= 0):
            raise ValueError("Positive finite sampling resolutions required")
        delta_yaw = (end[3] - start[3] + np.pi) % (2 * np.pi) - np.pi
        intervals = max(1, int(np.ceil(np.linalg.norm(end[:3] - start[:3]) / max_translation_step_m)),
                        int(np.ceil(abs(delta_yaw) / max_yaw_step_rad)))
        if intervals > 100_000:
            raise ValueError("Requested sweep exceeds bounded sampling size")
        samples = []
        for t in np.linspace(0., 1., intervals + 1):
            pose = start[:3] + t * (end[:3] - start[:3])
            sample = self.probe_upright_pose(xy_m=pose[:2], support_z_m=float(pose[2]),
                yaw_rad=float(start[3] + t * delta_yaw), envelope=envelope,
                support_tolerance_m=support_tolerance_m, max_slope_deg=max_slope_deg)
            samples.append(sample)
        failed = [i for i, sample in enumerate(samples) if not sample["sampled_reference_pass"]]
        return {"sampled_reference_pass": not failed, "sample_count": len(samples),
            "failed_sample_indices": failed, "first_failure": samples[failed[0]] if failed else None,
            "actual_translation_step_m": float(np.linalg.norm(end[:3] - start[:3]) / intervals),
            "actual_yaw_step_rad": abs(float(delta_yaw)) / intervals,
            "hardware_reachability_certified": False}
