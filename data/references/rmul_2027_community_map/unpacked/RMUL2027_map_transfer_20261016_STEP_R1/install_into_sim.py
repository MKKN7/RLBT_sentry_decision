"""Install a verified map-only package into the existing sentry simulation.

No network, Git changes, build, process control or legacy-map deletion.
Every replaced file is backed up. Unknown launch layouts are rejected before writing.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import ast


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patched_launch(text):
    ast.parse(text)
    old = "if world not in ('sentry_lab', 'depot'):"
    prior = "if world not in ('sentry_lab', 'depot', 'rmul_2027'):"
    new = "if world not in ('sentry_lab', 'depot', 'rmul_2027', 'rmul_2027_native'):"
    if old not in text and prior not in text and new not in text:
        raise ValueError('Unsupported launch world validation; manually merge the reference launch file. No files changed.')
    if 'def visual_suffix(context):' not in text:
        raise ValueError('This package expects the existing cad/preview launch profile. No files changed.')
    text = text.replace(old, new).replace(prior,new).replace(
        "raise ValueError('world must be sentry_lab or depot')",
        "raise ValueError('world must be sentry_lab, depot, rmul_2027 or rmul_2027_native')").replace(
        "raise ValueError('world must be sentry_lab, depot or rmul_2027')",
        "raise ValueError('world must be sentry_lab, depot, rmul_2027 or rmul_2027_native')")
    pattern = r"DeclareLaunchArgument\('world',\s*default_value='(?:sentry_lab|depot|rmul_2027)'\)"
    text, count = re.subn(pattern, "DeclareLaunchArgument('world',default_value='rmul_2027')", text)
    if count != 1:
        raise ValueError('Unsupported default world declaration. No files changed.')
    ast.parse(text)
    return text


def install(bundle, repository, check=False):
    import sys
    sys.path.insert(0, str(bundle))
    from verify_package import verify
    verify(bundle)
    repo = repository.expanduser().resolve()
    package = repo/'src/guodong_sentry_sim'
    launch = package/'launch/sentry_sim.launch.py'
    for rel in ['package.xml', 'CMakeLists.txt', 'launch/sentry_sim.launch.py',
                'models/guodong_sentry/model.sdf', 'models/guodong_sentry_preview/model.sdf',
                'worlds/depot.sdf', 'worlds/sentry_lab.sdf']:
        if not (package/rel).is_file():
            raise ValueError(f'Expected the complete existing guodong_sentry_sim with cad/preview profiles; missing {package/rel}')
    if repo == bundle.resolve() or repo.is_relative_to(bundle.resolve()):
        raise ValueError('Choose your existing full simulation repository, not this map-only bundle.')
    cmake = (package/'CMakeLists.txt').read_text(encoding='utf-8')
    directory_installs = re.findall(r'install\s*\(DIRECTORY\s+([^)]*)\)', cmake, re.S)
    installed_dirs = set()
    for line in directory_installs:
        installed_dirs.update(line.split('DESTINATION')[0].split())
    if not {'models', 'maps', 'worlds', 'launch'}.issubset(installed_dirs):
        raise ValueError('CMake must install the complete models/maps/worlds/launch directories. No files changed.')
    new_launch = patched_launch(launch.read_text(encoding='utf-8'))
    copies = {}
    payload = bundle/'src/guodong_sentry_sim'
    for directory in ['models/rmul_2027', 'maps/rmul_2027']:
        for p in (payload/directory).rglob('*'):
            if p.is_file():
                copies[Path('src/guodong_sentry_sim')/p.relative_to(payload)] = p
    for path in (payload/'worlds').glob('rmul_2027*.sdf'):
        copies[Path('src/guodong_sentry_sim/worlds')/path.name] = path
    for name in ['rmul_field.py', 'test_rmul_field.py', 'check_rmul_runtime.py', 'step_field.py', 'test_step_field.py']:
        copies[Path('tools')/name] = bundle/'tools'/name
    copies[Path('docs/RMUL_2027.md')] = bundle/'docs/RMUL_2027.md'
    copies[Path('docs/RMUL_2027_STEP_AUDIT.md')] = bundle/'docs/RMUL_2027_STEP_AUDIT.md'
    copies[Path('docs/validation/rmul_2027_source_machine.json')] = bundle/'docs/validation/source_machine.json'
    copies[Path('scripts/rmul_2027.sh')] = bundle/'scripts/run_integrated.sh'
    # The portable launcher is copied into the target repository's scripts/ directory.
    touched = list(copies)+[Path('src/guodong_sentry_sim/launch/sentry_sim.launch.py')]
    has_manifest = (repo/'manifest.json').is_file()
    manifest = None
    if has_manifest:
        manifest = json.loads((repo/'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest.get('files'), dict):
            raise ValueError('Unsupported existing manifest. No files changed.')
        touched.append(Path('manifest.json'))
    for rel in touched:
        if not (repo/rel).resolve().is_relative_to(repo):
            raise ValueError(f'Destination leaves the intended repository: {rel}')
    old_maps = {}
    for folder in [package/'worlds', package/'maps']:
        for p in folder.rglob('*'):
            if p.is_file() and not p.relative_to(folder).as_posix().startswith('rmul_2027'):
                old_maps[p.relative_to(repo).as_posix()] = sha(p)
    if check:
        return {'mode': 'check', 'repository': str(repo), 'files_to_install': len(copies),
                'legacy_assets_to_preserve': len(old_maps), 'compatible': True}
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = repo/'output/validation/rmul_transfer_backups'/stamp
    if not backup.resolve().is_relative_to(repo):
        raise ValueError('Backup directory leaves the intended repository')
    backup.mkdir(parents=True, exist_ok=False)
    previous = {}
    for rel in touched:
        path = repo/rel
        previous[rel.as_posix()] = path.is_file()
        if path.is_file():
            dest = backup/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    record = {'repository': str(repo), 'backup': str(backup), 'previously_existed': previous,
              'legacy_assets_sha256': old_maps, 'result': 'installing',
              'next': 'Build and test on this computer. Source-machine results are historical only.'}
    (backup/'INSTALL_RECORD.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    for rel, source in copies.items():
        dest = repo/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    launch.write_text(new_launch, encoding='utf-8', newline='\n')
    if manifest is not None:
        for rel in touched:
            if rel == Path('manifest.json'):
                continue
            p = repo/rel
            manifest['files'][rel.as_posix()] = {'sha256': sha(p), 'bytes': p.stat().st_size}
        manifest['updated_checkpoint'] = 'rmul_2027_map_transfer'
        (repo/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    for rel, old_sha in old_maps.items():
        if sha(repo/rel) != old_sha:
            raise RuntimeError(f'Legacy asset changed: {rel}; backup: {backup}')
    record['result'] = 'PASS'
    (backup/'INSTALL_RECORD.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Existing complete simulation repository root')
    parser.add_argument('--check', action='store_true', help='Read-only compatibility check')
    args = parser.parse_args()
    try:
        print(json.dumps(install(Path(__file__).resolve().parent, args.repo, args.check), indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, str(exc)+'\n')
