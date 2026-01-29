#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialize a pypsa-nza-data workspace.

Copies small, manually-prepared static inputs (nodes/links/lines tables)
from the installed package into the user's workspace.

Created on Thu Jan 29 15:35:06 2026
Phillippe Bruneau

"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import importlib.resources as resources

MANUAL_FILES = ["nodes.csv", "links_data.csv", "lines_data.csv"]


def _pkg_manual_dir() -> resources.abc.Traversable:
    return resources.files("pypsa_nza_data").joinpath("resources", "manual")


def copy_manual_files(dest_dir: Path, force: bool = False) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pkg_dir = _pkg_manual_dir()

    copied = 0
    skipped = 0

    for name in MANUAL_FILES:
        src = pkg_dir.joinpath(name)
        if not src.is_file():
            raise FileNotFoundError(f"Packaged manual file missing: {src}")

        dest = dest_dir / name
        if dest.exists() and not force:
            skipped += 1
            continue

        with resources.as_file(src) as src_path:
            shutil.copy2(src_path, dest)
        copied += 1

    print(f"Manual workspace directory: {dest_dir}")
    print(f"Copied: {copied} file(s); skipped: {skipped} file(s).")
    if skipped and not force:
        print("Tip: re-run with --force to overwrite existing files.")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="nza_init_workspace",
        description="Initialize a pypsa-nza-data workspace with packaged manual inputs.",
    )
    parser.add_argument("--root", required=True, help="Workspace root directory")
    parser.add_argument(
        "--manual-dir",
        default="data/manual",
        help="Relative path under --root for manual inputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manual files in the workspace",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    manual_dir = (root / args.manual_dir).resolve()

    try:
        copy_manual_files(manual_dir, force=args.force)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
