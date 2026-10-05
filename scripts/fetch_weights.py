#!/usr/bin/env python3
"""Download the trained HierarchiRetina weights from the v1.0 GitHub release.

Each file is saved where the notebooks and the package expect it, for example
outputs/stage3/checkpoints/best_fold0.pt, and its SHA-256 is checked against
scripts/weights_manifest.json. Files that are already present and correct are skipped.
Only the Python standard library is needed.

Usage
  python scripts/fetch_weights.py               # all 12 files
  python scripts/fetch_weights.py --stage 3     # only Stage III (LG-DRG)
  python scripts/fetch_weights.py --list        # show the files, sizes and target paths
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "weights_manifest.json"
BASE_URL = "https://github.com/asifuddin01/HierarchiRetina/releases/download/v1.0/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f"\r    {done / 1e9:5.2f} / {size / 1e9:.2f} GB", end="", flush=True)
    print()
    part.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["1", "2", "3"], help="download one stage only")
    ap.add_argument("--root", default=str(ROOT), help="repository root (default: this checkout)")
    ap.add_argument("--list", action="store_true", help="list the files and exit")
    args = ap.parse_args()

    rows = json.loads(MANIFEST.read_text())
    if args.stage:
        rows = [r for r in rows if r["asset"].startswith(f"stage{args.stage}_")]
    total = sum(r["bytes"] for r in rows)
    if args.list:
        for r in rows:
            print(f"{r['asset']:<44} {r['bytes'] / 1e9:5.2f} GB -> {r['place_at']}")
        print(f"{len(rows)} files, {total / 1e9:.2f} GB")
        return 0

    root = Path(args.root)
    print(f"{len(rows)} files, {total / 1e9:.2f} GB, from {BASE_URL}")
    bad = 0
    for r in rows:
        dest = root / r["place_at"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == r["bytes"] and sha256(dest) == r["sha256"]:
            print(f"  ok (already present)  {r['place_at']}")
            continue
        print(f"  downloading {r['asset']} -> {r['place_at']}")
        download(BASE_URL + r["asset"], dest, r["bytes"])
        if sha256(dest) != r["sha256"]:
            print(f"  CHECKSUM MISMATCH for {dest}; delete it and run again")
            bad += 1
        else:
            print("    SHA-256 verified")
    print("done" if bad == 0 else f"{bad} file(s) failed verification")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
