# -*- coding: utf-8 -*-

"""
Run the PyPSA-NZA data acquisition pipeline (static + dynamic loaders).

Design goals:
- Work from any working directory (reviewer-proof).
- Use the current environment's Python interpreter.
- Invoke loaders as installed modules (`python -m ...`), not as loose scripts.
- Support --dry-run and optional config overrides for each loader.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional


DYNAMIC_MODULE = "pypsa_nza_data.loaders.nza_load_dynamic_data_from_url"
STATIC_MODULE = "pypsa_nza_data.loaders.nza_load_static_data_from_url"


def _run_module(module: str, args: Optional[List[str]] = None) -> None:
    """
    Run a Python module using the current interpreter.
    Raises CalledProcessError on failure.
    """
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nza_run_loader_pipeline",
        description="Run static and dynamic data loaders in sequence.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run through to both loaders.",
    )

    p.add_argument(
        "--skip-dynamic",
        action="store_true",
        help="Skip the dynamic loader step.",
    )

    p.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip the static loader step.",
    )

    p.add_argument(
        "--dynamic-config",
        type=str,
        default=None,
        help="Path to dynamic loader YAML config (passed as --config).",
    )

    p.add_argument(
        "--static-config",
        type=str,
        default=None,
        help="Path to static loader YAML config (passed as --config).",
    )

    # Optional: allow users/reviewers to pass through extra args safely
    p.add_argument(
        "--dynamic-args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to the dynamic loader after a '--' separator.",
    )

    p.add_argument(
        "--static-args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to the static loader after a '--' separator.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    dynamic_cli: List[str] = []
    static_cli: List[str] = []

    if args.dry_run:
        dynamic_cli.append("--dry-run")
        static_cli.append("--dry-run")

    if args.dynamic_config:
        dynamic_cli.extend(["--config", args.dynamic_config])

    if args.static_config:
        static_cli.extend(["--config", args.static_config])

    # Pass-through arguments (support both None and empty list)
    if args.dynamic_args:
        dynamic_cli.extend(args.dynamic_args)

    if args.static_args:
        static_cli.extend(args.static_args)

    # Run steps
    if not args.skip_dynamic:
        _run_module(DYNAMIC_MODULE, dynamic_cli)

    if not args.skip_static:
        _run_module(STATIC_MODULE, static_cli)


if __name__ == "__main__":
    main()







# -----------------------------------------------------------------------------
"""
Created on Sun Dec 21 10:49:54 2025

@author: OEM
"""

# run_loader_pipeline.py
# import subprocess
# scripts = ["nza_load_dynamic_data_from_url.py", 
#            "nza_load_static_data_from_url.py"]

# for script in scripts:
#     subprocess.run(["python", script], check=True)
    
    
    
# # -*- coding: utf-8 -*-    