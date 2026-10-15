"""Recompute hashes for both release manifests after content edits.

Only refreshes hash VALUES for entries already listed; never adds or drops entries.
Run from the package root:  python tools/reseal_release.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path, text):
    """Write with LF line endings, regardless of host platform."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    provenance = ROOT / "provenance" / "release_source_hashes.json"
    sums = ROOT / "SHA256SUMS"

    changed_prov = []
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    for name, old in prov.items():
        p = ROOT / name
        if not p.is_file():
            raise SystemExit(f"Missing provenance source file: {name}")
        new = digest(p)
        if new != old:
            changed_prov.append(name)
            prov[name] = new
    write_text(provenance, json.dumps(prov, indent=2, ensure_ascii=False) + "\n")
    print(f"provenance: {len(prov)} entries, updated {len(changed_prov)}")
    for n in changed_prov:
        print(f"   updated {n}")

    lines = sums.read_text(encoding="utf-8").splitlines()
    changed_sums = []
    out = []
    for line in lines:
        if not line.strip():
            continue
        old_hash, name = line.split("  ", 1)
        name = name.strip()
        p = ROOT / name
        if not p.is_file():
            raise SystemExit(f"Missing SHA256SUMS file: {name}")
        new_hash = digest(p)
        if new_hash != old_hash:
            changed_sums.append(name)
        out.append(f"{new_hash}  {name}")
    write_text(sums, "\n".join(out) + "\n")
    print(f"SHA256SUMS: {len(out)} entries, updated {len(changed_sums)}")
    for n in changed_sums:
        print(f"   updated {n}")


if __name__ == "__main__":
    main()
