"""Verify extracted migration payload with Python's standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'rmul-map-transfer-v1':
        raise ValueError('Unsupported package format')
    for rel, entry in manifest['files'].items():
        path = (root/rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing/unsafe file: {rel}')
        if path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'File changed: {rel}')
    return len(manifest['files'])


if __name__ == '__main__':
    print(f'PASS: {verify()} package files verified (SHA-256).')
