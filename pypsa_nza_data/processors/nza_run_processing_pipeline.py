# -*- coding: utf-8 -*-
"""
nza_run_processing_pipeline.py

Orchestrates the Step 2 processing pipeline (dynamic + static processing and
downstream transforms) using a loader-style CLI.

Key design goals:
- Run as an installed module: `python -m pypsa_nza_data.processors.nza_run_processing_pipeline ...`
- Call sub-steps via `sys.executable -m ...` so it works from any directory and on Windows/Linux
- Keep YAML per-script (Option 1). Defaults are packaged under `pypsa_nza_data/config/`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    # Python 3.9+
    from importlib.resources import files as pkg_files
except Exception:  # pragma: no cover
    pkg_files = None


PACKAGE_NAME = "pypsa_nza_data"

# Default config basenames (Option 1: separate YAML per processor script)
DEFAULT_DYNAMIC_CONFIG = "nza_process_dynamic_data.yaml"
DEFAULT_STATIC_CONFIG = "nza_process_static_data.yaml"
DEFAULT_LOAD_PROFILE_CONFIG = "nza_cx_config.yaml"
DEFAULT_CONVERT_CONFIG = "nza_convert_config.yaml"


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    config_path: Optional[Path] = None


def _default_config_path(basename: str) -> Path:
    """
    Resolve packaged default YAML config path.
    Expects configs in: pypsa_nza_data/config/<basename>
    """
    if pkg_files is None:
        raise RuntimeError("importlib.resources.files not available; please use Python >= 3.9")

    candidate = pkg_files(PACKAGE_NAME).joinpath(f"config/{basename}")
    return Path(str(candidate))


def _run_module(module: str, config: Optional[Path], dry_run: bool, verbose: bool) -> int:
    """
    Execute a sub-step as: sys.executable -m <module> [--config <config>]
    """
    cmd = [sys.executable, "-m", module]
    if config is not None:
        cmd += ["--config", str(config)]

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return 0

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"? Step failed: {module} (exit code {e.returncode})", file=sys.stderr)
        return e.returncode


def build_steps(args: argparse.Namespace) -> list[Step]:
    """
    Construct the ordered list of processing steps based on CLI flags.
    """
    # Decide which blocks are enabled
    if not any([args.dynamic, args.static, args.load_profile, args.convert, args.all]):
        # default: run all
        run_dynamic = run_static = run_load = run_convert = True
    else:
        run_dynamic = args.all or args.dynamic
        run_static = args.all or args.static
        run_load = args.all or args.load_profile
        run_convert = args.all or args.convert

    dyn_cfg = Path(args.config_dynamic) if args.config_dynamic else _default_config_path(DEFAULT_DYNAMIC_CONFIG)
    sta_cfg = Path(args.config_static) if args.config_static else _default_config_path(DEFAULT_STATIC_CONFIG)
    load_cfg = Path(args.config_load) if args.config_load else _default_config_path(DEFAULT_LOAD_PROFILE_CONFIG)
    conv_cfg = Path(args.config_convert) if args.config_convert else _default_config_path(DEFAULT_CONVERT_CONFIG)

    steps: list[Step] = []

    # Order: dynamic processing ? static processing ? load profile ? conversions
    # (You can adjust ordering later if your actual dependencies differ.)
    if run_dynamic:
        steps.append(
            Step(
                name="process_dynamic",
                module="pypsa_nza_data.processors.nza_process_dynamic_data",
                config_path=dyn_cfg,
            )
        )

    if run_static:
        steps.append(
            Step(
                name="process_static",
                module="pypsa_nza_data.processors.nza_process_static_data",
                config_path=sta_cfg,
            )
        )

    if run_load:
        steps.append(
            Step(
                name="create_load_profile",
                module="pypsa_nza_data.processors.nza_create_load_profile",
                config_path=load_cfg,
            )
        )

    if run_convert:
        steps.append(
            Step(
                name="convert_energy_to_power",
                module="pypsa_nza_data.processors.nza_convert_energy_to_power",
                config_path=conv_cfg,
            )
        )

    return steps


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nza_run_processing_pipeline",
        description="Run the PyPSA-NZA Step 2 processing pipeline (dynamic/static processing and transforms).",
    )

    # Mode selection (similar to loaders)
    g = p.add_argument_group("Pipeline selection")
    g.add_argument("--all", action="store_true", help="Run all processing steps (default if no flags are given).")
    g.add_argument("--dynamic", action="store_true", help="Run dynamic processing step only.")
    g.add_argument("--static", action="store_true", help="Run static processing step only.")
    g.add_argument("--load-profile", action="store_true", help="Run load profile creation step only.")
    g.add_argument("--convert", action="store_true", help="Run energy?power conversion step only.")

    # Config overrides (Option 1)
    c = p.add_argument_group("Config overrides (optional)")
    c.add_argument("--config-dynamic", type=str, default=None, help="Override YAML for dynamic processing.")
    c.add_argument("--config-static", type=str, default=None, help="Override YAML for static processing.")
    c.add_argument("--config-load", type=str, default=None, help="Override YAML for load profile step.")
    c.add_argument("--config-convert", type=str, default=None, help="Override YAML for conversion step.")

    # Execution flags
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument("--verbose", action="store_true", help="Print executed commands and additional info.")

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    steps = build_steps(args)
    if not steps:
        print("? No steps selected.")
        return 2

    print("================================================================================")
    print("                     PYPSA-NZA PROCESSING PIPELINE (STEP 2)")
    print("================================================================================")
    print(f"Selected steps: {', '.join(s.name for s in steps)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print("")

    for s in steps:
        # Defensive: verify config exists (packaged configs will exist if installed correctly)
        if s.config_path is not None and not s.config_path.exists():
            print(f"? Missing config for step '{s.name}': {s.config_path}", file=sys.stderr)
            return 2

        rc = _run_module(s.module, s.config_path, dry_run=args.dry_run, verbose=args.verbose)
        if rc != 0:
            return rc

    print("")
    print("? Processing pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
