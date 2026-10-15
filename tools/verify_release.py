"""Verify packaged code/config/weight hashes without importing Torch or using a GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    models = json.loads((ROOT / 'models.json').read_text())
    for name, entry in models.items():
        for field, sha in [('weight', 'sha256'), ('config', 'config_sha256')]:
            path = ROOT / entry[field]
            if not path.is_file():
                raise SystemExit(f'Missing {entry[field]}; place the corresponding Release asset here.')
            if digest(path) != entry[sha]:
                raise SystemExit(f'Checksum mismatch: {entry[field]}')
        print(f'PASS {name}: configuration and tensor-only weight archive identity')
    for metadata in sorted((ROOT/'adapters').glob('*.json')):
        entry = json.loads(metadata.read_text())
        asset = metadata.with_suffix('.pth')
        if not asset.is_file() or digest(asset) != entry['sha256']:
            raise SystemExit(f'Missing or mismatched adapter snapshot: {asset.name}')
        if models[entry['parent_model']]['sha256'] != entry['parent_sha256']:
            raise SystemExit(f'Adapter/source mismatch: {asset.name}')
        print(f'PASS {asset.name}: tensor snapshot identity and source binding')
    hashes = json.loads((ROOT / 'provenance/release_source_hashes.json').read_text())
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise SystemExit(f'Source checksum mismatch: {name}')
    print(f'PASS {len(hashes)} packaged runtime source files')
    print('This verifies identity; dataset AP and deserialization are separate checks.')


if __name__ == '__main__':
    main()
