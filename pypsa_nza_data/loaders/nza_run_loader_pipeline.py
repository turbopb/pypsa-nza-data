#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pypsa_nza_data.loaders.nza_run_loader_pipeline

Orchestrate the loader stage by running:
- pypsa_nza_data.loaders.nza_load_dynamic_data_from_url
- pypsa_nza_data.loaders.nza_load_static_data_from_url

Design contract (reviewer-safe):
- --root is REQUIRED and defines the workspace directory.
- Config paths are independent of --root.
- All downloaded data and logs are written under --root by the underlying loaders.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
import importlib.resources as resources


DEFAULT_DYNAMIC_CONFIG = "nza_load_dynamic_data.yaml"
DEFAULT_STATIC_CONFIG = "nza_load_static_data.yaml"


def _default_config_path(basename: str) -> Path:
    return resources.files("pypsa_nza_data").joinpath("config", basename)


def _build_cmd(module: str, config_path: Path, root: Path, extra_args: list[str] | None, dry_run: bool) -> list[str]:
    cmd = [sys.executable, "-m", module, "--config", str(config_path), "--root", str(root)]
    if dry_run:
        cmd.append("--dry-run")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="nza_run_loader_pipeline",
        description="Run pypsa-nza-data loader pipeline (dynamic + static)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--root", type=str, required=True, help="Workspace root directory passed to loader scripts")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run through to loader scripts")

    parser.add_argument("--skip-dynamic", action="store_true", help="Skip the dynamic loader")
    parser.add_argument("--skip-static", action="store_true", help="Skip the static loader")

    parser.add_argument("--dynamic-config", type=str, default=None, help="Path to dynamic loader YAML (defaults to packaged config)")
    parser.add_argument("--static-config", type=str, default=None, help="Path to static loader YAML (defaults to packaged config)")

    # Allow passing additional args to sub-scripts without the pipeline needing to know them.
    # Example:
    #   --dynamic-args --dataset generation --dataset demand
    parser.add_argument("--dynamic-args", nargs=argparse.REMAINDER, default=None, help="Extra args for dynamic loader (everything after this flag)")
    parser.add_argument("--static-args", nargs=argparse.REMAINDER, default=None, help="Extra args for static loader (everything after this flag)")

    args = parser.parse_args()

    workspace_root = Path(args.root).expanduser().resolve()

    dyn_cfg = Path(args.dynamic_config).expanduser().resolve() if args.dynamic_config else _default_config_path(DEFAULT_DYNAMIC_CONFIG)
    sta_cfg = Path(args.static_config).expanduser().resolve() if args.static_config else _default_config_path(DEFAULT_STATIC_CONFIG)

    if not args.skip_dynamic:
        cmd = _build_cmd(
            "pypsa_nza_data.loaders.nza_load_dynamic_data_from_url",
            dyn_cfg,
            workspace_root,
            args.dynamic_args,
            args.dry_run,
        )
        print("\n" + "=" * 80)
        print("RUNNING: dynamic loader")
        print("=" * 80)
        print("Command:")
        print("  " + " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)

    if not args.skip_static:
        cmd = _build_cmd(
            "pypsa_nza_data.loaders.nza_load_static_data_from_url",
            sta_cfg,
            workspace_root,
            args.static_args,
            args.dry_run,
        )
        print("\n" + "=" * 80)
        print("RUNNING: static loader")
        print("=" * 80)
        print("Command:")
        print("  " + " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)

    print("\n" + "=" * 80)
    print("LOADER PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Workspace root: {workspace_root}")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
