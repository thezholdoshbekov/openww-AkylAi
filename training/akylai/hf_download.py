"""Download the AkylAi wake-word data from the Hugging Face Hub
(repo: the-cramer-project/Wake_word_Akylai).

The TOKEN is read from the HF_TOKEN environment variable so it never ends up in
the script or in your shell history file.

PowerShell:
    $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxx"     # https://huggingface.co/settings/tokens

    # 1) see what is in the repo (find the ~700 real positives):
    python training/akylai/hf_download.py --list

    # 2) download only what you need (glob over repo paths), e.g.:
    python training/akylai/hf_download.py --pattern "*real*" "*positive*"

    # 3) or grab the whole repo:
    python training/akylai/hf_download.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Simonlob/Wake_word_Akylai"
REPO_TYPE = "dataset"                       # it's a dataset repo
DEST = Path(r"D:/akylai_oww/data_hf")       # where downloads land (D: has free space)


def get_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        sys.exit('No token found. In PowerShell run:  $env:HF_TOKEN = "hf_..."  then re-run.')
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list every file in the repo and exit")
    ap.add_argument("--pattern", nargs="*", default=None,
                    help='allow-patterns (globs over repo paths), e.g. "*real*" "*.parquet"')
    ap.add_argument("--all", action="store_true", help="download the entire repo")
    args = ap.parse_args()

    token = get_token()
    api = HfApi(token=token)

    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    print(f"repo {REPO_ID} ({REPO_TYPE}) — {len(files)} files\n")

    if args.list or (not args.all and not args.pattern):
        # group by top-level folder so the layout is easy to read
        import collections
        by_dir: dict[str, int] = collections.Counter()
        for f in files:
            top = f.split("/")[0] if "/" in f else "(root)"
            by_dir[top] += 1
        print("top-level layout (folder: #files):")
        for d, n in sorted(by_dir.items()):
            print(f"  {d}: {n}")
        print("\nfirst 60 paths:")
        for f in sorted(files)[:60]:
            print("  ", f)
        if not args.list:
            print("\n(showing layout only — re-run with --pattern \"...\" or --all to download)")
        return

    allow = args.pattern if args.pattern else None
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"downloading -> {DEST}\n  allow_patterns = {allow or 'ALL'}")
    path = snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, token=token,
        local_dir=str(DEST), allow_patterns=allow,
    )
    print(f"\nDone -> {path}")


if __name__ == "__main__":
    main()
