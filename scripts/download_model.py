#!/usr/bin/env python3
"""Download a supported model at its validated revision and verify every large file's sha256.

    python scripts/download_model.py --model clef-flash --dir ./models/clef-flash

Resumable: re-run after an interruption. Honors HF_ENDPOINT (mirrors) and HTTPS_PROXY.
Needs only ``huggingface_hub`` (``pip install huggingface_hub``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# Kept here instead of importing the package so the script runs before installation.
MODELS = {
    "clef-flash": ("Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="clef-flash", choices=sorted(MODELS))
    parser.add_argument("--dir", required=True, help="target directory")
    parser.add_argument("--revision", help="override the validated revision (unsupported)")
    args = parser.parse_args()
    repo_id, revision = MODELS[args.model]
    revision = args.revision or revision

    print(f"downloading {repo_id}@{revision[:7]} to {args.dir}")
    snapshot_download(repo_id, revision=revision, local_dir=args.dir, max_workers=8)

    print("verifying sha256 of LFS files")
    failed = 0
    for entry in HfApi().list_repo_tree(repo_id, revision=revision, recursive=True):
        lfs = getattr(entry, "lfs", None)
        if not lfs:
            continue
        ok = sha256(Path(args.dir) / entry.path) == lfs.sha256
        failed += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {entry.path}")
    if failed:
        print(f"{failed} file(s) failed verification; delete them and re-run", file=sys.stderr)
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
