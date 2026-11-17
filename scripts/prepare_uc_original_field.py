#!/usr/bin/env python3
"""Prepare the original UC STL with the existing, reviewed contact converter.

Only a private container directory is written. The source project, its current
field symlink, robot, and existing generated field are checked before/after.
This prepares resources; it does not run ROS, navigation, or physics validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "/tmp/rl_bt_nav_probe/fields/rmuc2027_original"
SOURCE_SHA256 = "2835c0e755c255f757d810d260283c623fbd4a49a8fd986487c70ad2a89e9d10"
TOOL_SHA256 = "70c72b5cd5b8d148d57a8b0e3b2421e45ab22db7aee684ee70acc46ef4a4bbe6"


CONTAINER_PROGRAM = r'''
from pathlib import Path
import hashlib, json, os, struct, subprocess, sys, time
import xml.etree.ElementTree as ET

options = json.loads(sys.argv[1])
source = Path('/opt/pb2025_nav_ws/src/pb2025_light_sim/meshes/rmuc_2027.stl')
tool = Path('/opt/mid360_mujoco/tools/prepare_field.py')
reference = Path('/opt/mid360_mujoco/generated/rmuc2027-v5')
reference_manifest = json.loads((reference/'manifest.json').read_text())
reference_world = Path(reference_manifest['world'])
root = Path('/tmp/rl_bt_nav_probe/fields')
out = Path(options['output'])
if not out.is_absolute() or out.parent.resolve() != root.resolve():
    raise ValueError('Output must be one direct child of the private fields directory')
if out.is_symlink() or root.is_symlink():
    raise ValueError('Private output/root must not be symlinks')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protected_state():
    paths = [source, tool, reference_world,
        Path('/opt/pb2025_nav_ws/src/pb2025_light_sim/meshes/rmuc_2027_driveable.stl'),
        Path('/opt/mid360_mujoco/configs/robot.yaml'),
        reference/'manifest.json', reference/'field.stl']
    return {'files_sha256': {str(p): digest(p) for p in paths},
            'field_current_resolved': str(Path('/opt/mid360_mujoco/generated/field-current').resolve())}

before = protected_state()
if before['files_sha256'][str(source)] != options['source_sha256']:
    raise ValueError('Original source hash differs from the reviewed source')
if before['files_sha256'][str(tool)] != options['tool_sha256']:
    raise ValueError('Preparation tool changed; review it before preparing')
collision = ET.parse(reference_world).find(".//model[@name='rmuc_2027']/link/collision")
pose_text = collision.findtext('pose')
pose = [float(v) for v in pose_text.split()]
if len(pose) != 6 or any(v != 0 for v in pose[3:]):
    raise ValueError('Nonzero/invalid source rotation requires explicit review')
if pose[:3] != reference_manifest['translation']:
    raise ValueError('World pose and existing field manifest disagree')

# Restrict this process and its child to two existing CPUs. Thread variables
# below also cap native numerical libraries; no CUDA or simulator is launched.
allowed_cpus = sorted(os.sched_getaffinity(0))[:2]
os.sched_setaffinity(0, allowed_cpus)
env = dict(os.environ)
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    env[key] = '2'
env['CUDA_VISIBLE_DEVICES'] = ''
os.environ.update({key: env[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'CUDA_VISIBLE_DEVICES')})

root.mkdir(parents=True, exist_ok=True)
world = root/(out.name+'_input.world')
log = root/(out.name+'.prepare.log')
started = time.monotonic()
reused = out.exists()
if reused:
    if not (out/'manifest.json').is_file():
        raise ValueError('Private output exists without a complete manifest; preserving it for diagnosis')
else:
    tree = ET.Element('sdf', version='1.7')
    w = ET.SubElement(tree, 'world', name='rl_bt_private_resource_preparation')
    model = ET.SubElement(w, 'model', name='rmuc_2027')
    link = ET.SubElement(model, 'link', name='field')
    c = ET.SubElement(link, 'collision', name='original_rmuc_2027_collision')
    ET.SubElement(c, 'pose').text = pose_text
    geometry = ET.SubElement(c, 'geometry')
    mesh = ET.SubElement(geometry, 'mesh')
    ET.SubElement(mesh, 'uri').text = str(source)
    new_world = ET.tostring(tree, encoding='utf-8', xml_declaration=True)
    if world.exists() and world.read_bytes() != new_world:
        raise ValueError('Private input world exists with different contents; refusing to replace it')
    if not world.exists():
        world.write_bytes(new_world)
    with log.open('x') as stream:
        command = [sys.executable, str(tool), str(world), str(out)]
        result = subprocess.run(command, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError('Reviewed converter failed, rc='+str(result.returncode)+', log='+str(log))

import numpy as np
dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
def stl_info(path):
    data = path.read_bytes()
    n = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84+n*50:
        raise ValueError('Not the expected binary STL: '+str(path))
    vertices = np.frombuffer(data, dtype=dtype, offset=84)['vertices'].astype(float)
    if not np.isfinite(vertices).all():
        raise ValueError('Nonfinite mesh vertices')
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'triangles':n,'bounds_m':[vertices.min(axis=(0,1)).tolist(),
                                       vertices.max(axis=(0,1)).tolist()]}, vertices

manifest_path = out/'manifest.json'
manifest = json.loads(manifest_path.read_text())
if (Path(manifest['source']).resolve() != source.resolve()
        or manifest['source_sha256'] != options['source_sha256']):
    raise ValueError('Prepared manifest source mismatch')
if manifest['translation'] != pose[:3] or manifest['world'] != str(world):
    raise ValueError('Prepared manifest transform/input world mismatch')
source_info, raw = stl_info(source)
prepared_info, prepared = stl_info(out/'field.stl')
if raw.shape != prepared.shape:
    raise ValueError('Source/prepared triangle shapes differ')
error = float(np.max(np.abs(prepared-(raw+np.asarray(pose[:3])))))
if error >= 3e-6:
    raise ValueError('Prepared mesh differs from translated original, max error='+str(error))
pieces = sorted((out/'collision').glob('*.obj'))
if {p.name for p in pieces} != {p['file'] for p in manifest['pieces']}:
    raise ValueError('Collision files differ from manifest')
if len(pieces) != manifest['collision_pieces']:
    raise ValueError('Collision count differs from manifest')
aggregate = hashlib.sha256()
for p in pieces:
    aggregate.update(p.name.encode()+b'\0')
    aggregate.update(hashlib.sha256(p.read_bytes()).digest())
visuals = sorted((out/'visual').glob('*.stl'))
visual_info = [stl_info(p)[0] for p in visuals]
if sum(v['triangles'] for v in visual_info) != prepared_info['triangles']:
    raise ValueError('Visual split triangle count mismatch')
after = protected_state()
if before != after:
    raise ValueError('Protected source project files changed during preparation')
files = [p for p in out.rglob('*') if p.is_file()]
report = {
    'status':'PASS_PRIVATE_ORIGINAL_UC_RESOURCE_PREPARATION',
    'prepared_field_dir':str(out), 'reused_existing_verified_output':reused,
    'source_mesh':source_info, 'prepared_mesh':prepared_info,
    'source_mesh_resolved_path':str(source.resolve()),
    'source_is_original_unsuffixed_rmuc_stl':True,
    'source_is_driveable_derivative':False,
    'translation_applied_once_m':pose[:3],
    'prepared_vs_translated_original_max_vertex_error_m':error,
    'source_triangle_order_preserved_within_3um':True,
    'converter':{'path':str(tool),'sha256':digest(tool),
        'interface':'python prepare_field.py private_input.world private_output',
        'implementation_modified':False,'physical_approximation_parameters_modified':False,
        'input_world':str(world),'input_world_sha256':digest(world),
        'input_pose_copied_from':str(reference_world),
        'log_path':str(log),'cpu_affinity':allowed_cpus,
        'maximum_native_library_threads':2,'gpu_enabled':False},
    'manifest_sha256':digest(manifest_path),
    'collision':{'count':len(pieces),'obj_bytes':sum(p.stat().st_size for p in pieces),
        'aggregate_sorted_name_and_sha256':aggregate.hexdigest(),
        'file_set_matches_manifest':True,'approximation_limitations':manifest['limitations'],
        'skipped_degenerate_area_m2':manifest['skipped_degenerate_area']},
    'visual_meshes':visual_info,
    'prepared_files':len(files),'prepared_total_file_bytes':sum(p.stat().st_size for p in files),
    'elapsed_prepare_and_verification_wall_s':time.monotonic()-started,
    'protected_source_state_before':before,'protected_source_state_after':after,
    'protected_source_files_unchanged':before==after,
    'field_current_modified':False,'robot_geometry_or_dynamics_modified':False,
    'ros_or_simulator_started':False,'physical_navigation_or_tunnel_test_passed':False,
    'official_referee_coordinate_registration_validated':False,
    'rfid_or_healing_semantics_inferred_from_mesh':False,
}
print(json.dumps(report))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="Combat_Sentry2026")
    parser.add_argument("--field-out", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "data/reports/uc_original_field_20261116.json")
    args = parser.parse_args()
    inspected = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", args.container],
        check=True, text=True, capture_output=True,
    )
    if inspected.stdout.strip() != "true":
        raise RuntimeError("The selected container is not running")
    print("Preparing/verifying original UC STL in a private container directory.", flush=True)
    options = {"output": args.field_out, "source_sha256": SOURCE_SHA256,
               "tool_sha256": TOOL_SHA256}
    result = subprocess.run(
        ["docker", "exec", args.container, "/opt/mid360_mujoco/.venv/bin/python",
         "-c", CONTAINER_PROGRAM, json.dumps(options)],
        check=False, text=True, capture_output=True,
    )
    if result.returncode:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"Private field preparation failed (exit {result.returncode})")
    report = json.loads(result.stdout)
    report["container"] = args.container
    report["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    report["host_rebuild_script"] = str(Path(__file__).resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.report),
        "prepared_field_dir": report["prepared_field_dir"],
        "prepared_mesh_sha256": report["prepared_mesh"]["sha256"],
        "triangles": report["prepared_mesh"]["triangles"],
        "collision_pieces": report["collision"]["count"],
        "translation_applied_once_m": report["translation_applied_once_m"],
        "physical_navigation_or_tunnel_test_passed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
