#!/usr/bin/env python3
"""Sample the saved UC reference mesh vertically without collapsing layers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_bt.layered_geometry import read_binary_stl, rasterize_vertical_surfaces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=ROOT / "assets/navigation_reference/field/field.stl")
    parser.add_argument("--resolution", type=float, default=.1)
    parser.add_argument("--label", default="UC reference")
    parser.add_argument("--out", type=Path, default=ROOT / "data/derived/uc_surface_layers_010m")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    mesh_hash = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    triangles = read_binary_stl(args.mesh)
    sampled = rasterize_vertical_surfaces(triangles, resolution=args.resolution)
    cache = args.out / "surfaces.npz"
    np.savez_compressed(cache, **sampled)
    counts = sampled["surface_count_xy"]
    low = sampled["low_slope_surface_candidate"]
    gap = sampled["z_to_next_surface_m"]
    low_gaps = gap[low & np.isfinite(gap)]
    report = {"artifact_type": "vertical_mesh_intersection_cache_not_navigation_acceptance",
        "map_label": args.label,
        "source": str(args.mesh.resolve()), "source_sha256": mesh_hash,
        "source_prepared_world_coordinates": True, "additional_translation_or_scaling_applied": False,
        "resolution_m": args.resolution, "grid_shape_xy": sampled["grid_shape_xy"].tolist(),
        "triangles": len(triangles), "sampled_surfaces": len(sampled["surface_z_m"]),
        "columns_total": counts.size, "columns_with_hits": int((counts > 0).sum()),
        "columns_with_multiple_surfaces": int((counts > 1).sum()),
        "max_surfaces_per_column": int(counts.max()),
        "low_slope_intersections": int(low.sum()),
        "vertical_or_degenerate_triangles": int(sampled["vertical_or_degenerate_face_count"]),
        "low_slope_with_finite_upper_intersection": len(low_gaps),
        "low_slope_vertical_gap_quantiles_m": dict(zip(["p10", "p50", "p90"],
            np.quantile(low_gaps, [.1, .5, .9]).tolist())) if len(low_gaps) else {},
        "cache": str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else str(cache),
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "official_zone_polygons_registered": False, "robot_envelope_verified": False,
        "limitations": ["Intersections have no ground/roof/interior semantic classification.",
            "No higher sampled surface is unknown/open, never infinite certified clearance.",
            "Vertical faces, thin objects and between-ray geometry require full mesh footprint tests.",
            "Surface spacing is not free-space certification; solid interior intervals remain possible.",
            "No ground connectivity, tunnel route, official zones or Nav2 integration inferred.",
            "Copied simulator world geometry; official field coordinate registration remains incomplete."],
        "wall_s": time.perf_counter() - started}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    shape = tuple(sampled["grid_shape_xy"])
    first = np.full(counts.size, np.nan)
    hit = counts.ravel() > 0
    first[hit] = sampled["surface_z_m"][sampled["column_offsets"][:-1][hit]]
    origin = sampled["origin_xy_m"]
    extent = [origin[0], origin[0] + shape[0]*args.resolution,
              origin[1], origin[1] + shape[1]*args.resolution]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
    a = axes[0].imshow(np.ma.masked_where(counts.T == 0, counts.T), origin="lower",
        extent=extent, interpolation="nearest", vmin=1, vmax=max(2, int(counts.max())), cmap="viridis")
    axes[0].set_title("Separate sampled surfaces per XY column")
    fig.colorbar(a, ax=axes[0], label="Intersections (blank = no hit)")
    a = axes[1].imshow(first.reshape(shape).T, origin="lower", extent=extent,
                       interpolation="nearest", cmap="terrain")
    axes[1].set_title("Lowest sampled intersection (not certified ground)")
    fig.colorbar(a, ax=axes[1], label="Prepared-world height (m)")
    for ax in axes:
        ax.set(xlabel="Prepared-world X (m)", ylabel="Prepared-world Y (m)")
    fig.suptitle(f"{args.label} mesh: vertical samples, not a traversability map")
    fig.savefig(args.out / "preview.png", dpi=140)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ["sampled_surfaces", "columns_with_multiple_surfaces",
                                            "max_surfaces_per_column", "wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
