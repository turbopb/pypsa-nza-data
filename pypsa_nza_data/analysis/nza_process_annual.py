# -*- coding: utf-8 -*-
"""
process_annual.py

Processes all 12 monthly export, import, or gen files for a given year,
aggregates POCs to site level, and concatenates into a single annual
time series for each.

Usage:
    python process_annual.py --year 2024 --outdir ./annual
    python process_annual.py --year 2024 --outdir ./annual --flow gen
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

from .nza_poc_aggregator import (
    load_poc_file,
    aggregate_by_site_all_voltages,
    energy_check
)

# --- Logging setup ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# --- Annual processor -------------------------------------------------------

def process_annual(base_dir: Path,
                   year: int,
                   flow: str,
                   outdir: Path,
                   threshold_kv: int = 110) -> pd.DataFrame:
    """
    Process all monthly files for a given year and flow direction.

    Parameters
    ----------
    base_dir : Path
        Base data directory containing year subfolders.
    year : int
        Year to process, e.g. 2024.
    flow : str
        'export', 'import', or 'gen'.
    outdir : Path
        Output directory.
    threshold_kv : int
        Voltage threshold for sub-threshold reporting only.

    Returns
    -------
    pd.DataFrame
        Annual time series, site-aggregated, all voltages combined.
    """
    flow_dir = base_dir / str(year) / flow / "cons_MWh"
    monthly_frames = []
    monthly_stats  = []

    for month in range(1, 13):
        filename = f"{year}{month:02d}_{flow}_md.csv"
        filepath = flow_dir / filename

        if not filepath.exists():
            log.warning(f"  File not found, skipping: {filepath}")
            continue

        log.info(f"Processing {filename}...")

        df_raw       = load_poc_file(filepath)
        df_sites_all = aggregate_by_site_all_voltages(df_raw)
        energy_check(df_raw, df_sites_all,
                     label=f"({year}-{month:02d})")

        monthly_frames.append(df_sites_all)

        # Collect monthly stats for summary table
        monthly_stats.append({
            "month"      : f"{year}-{month:02d}",
            "timesteps"  : len(df_raw),
            "pocs"       : len(df_raw.columns),
            "sites"      : len(df_sites_all.columns),
            "energy_GWh" : df_sites_all.sum().sum() / 1000.0
        })

    if not monthly_frames:
        raise RuntimeError(f"No files found for {year} {flow}")

    # Concatenate all months into single annual time series
    log.info(f"Concatenating {len(monthly_frames)} months...")
    df_annual = pd.concat(monthly_frames, axis=0)

    # Check for duplicate timestamps
    n_dupes = df_annual.index.duplicated().sum()
    if n_dupes > 0:
        log.warning(f"  {n_dupes} duplicate timestamps found -- check overlap")
    else:
        log.info(f"  No duplicate timestamps")

    # Check for gaps
    expected_freq = pd.tseries.frequencies.to_offset("30min")
    full_index    = pd.date_range(
        start=df_annual.index[0],
        end=df_annual.index[-1],
        freq=expected_freq
    )
    n_gaps = len(full_index) - len(df_annual)
    if n_gaps > 0:
        log.warning(f"  {n_gaps} missing timesteps in annual series")
    else:
        log.info(f"  Time series is complete with no gaps")

    # --- Clean annual time series -------------------------------------------
    # Fill NaN with zero -- site absent in a month means zero generation
    n_nan_before = df_annual.isna().sum().sum()
    df_annual = df_annual.fillna(0.0)
    log.info(f"  Filled {n_nan_before} NaN values with zero")

    # Drop columns that are all zero -- registered but never active
    all_zero = df_annual.columns[df_annual.sum() == 0].tolist()
    if all_zero:
        log.info(f"  Dropping {len(all_zero)} sites with no data: {all_zero}")
        df_annual = df_annual.drop(columns=all_zero)
    else:
        log.info(f"  No all-zero sites found")

    # Write annual output
    annual_path = outdir / f"{year}_{flow}_sites_all.csv"
    df_annual.to_csv(annual_path)
    log.info(f"  Annual {flow} written to: {annual_path}")

    # Write monthly summary
    stats_df   = pd.DataFrame(monthly_stats)
    stats_path = outdir / f"{year}_{flow}_monthly_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    log.info(f"  Monthly summary written to: {stats_path}")

    # Print summary table
    log.info(f"=== Annual {flow} summary ===")
    log.info(f"  {'Month':<12} {'Timesteps':>10} "
             f"{'POCs':>6} {'Sites':>6} {'Energy GWh':>12}")
    for row in monthly_stats:
        log.info(f"  {row['month']:<12} {row['timesteps']:>10} "
                 f"{row['pocs']:>6} {row['sites']:>6} "
                 f"{row['energy_GWh']:>12.1f}")
    log.info(f"  {'TOTAL':<12} {sum(r['timesteps'] for r in monthly_stats):>10} "
             f"{'':>6} {'':>6} "
             f"{sum(r['energy_GWh'] for r in monthly_stats):>12.1f}")

    return df_annual


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process full year of export, import, or gen POC files."
    )
    parser.add_argument("--year",      type=int, required=True,
                        help="Year to process, e.g. 2024")
    parser.add_argument("--basedir",   default=".",
                        help="Base data directory (default: current)")
    parser.add_argument("--outdir",    default="./annual",
                        help="Output directory (default: ./annual)")
    parser.add_argument("--threshold", type=int, default=110,
                        help="Voltage threshold for reporting (default: 110)")
    parser.add_argument("--flow",      default=None,
                        help="Process single flow only: export, import, or gen. "
                             "If omitted, processes export and import and "
                             "performs cross-check.")
    args = parser.parse_args()

    base_dir = Path(args.basedir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine which flows to process
    if args.flow:
        flows = [args.flow]
    else:
        flows = ["export", "import"]

    # Process each flow
    results = {}
    for flow in flows:
        log.info(f"{'='*60}")
        log.info(f"Processing {args.year} {flow.upper()}...")
        log.info(f"{'='*60}")
        results[flow] = process_annual(
            base_dir, args.year, flow, outdir, args.threshold
        )

    # Cross-check only when both export and import are processed together
    if "export" in results and "import" in results:
        total_export = results["export"].sum().sum() / 1000.0
        total_import = results["import"].sum().sum() / 1000.0
        losses       = total_import - total_export
        losses_pct   = losses / total_import * 100

        log.info(f"{'='*60}")
        log.info(f"Annual energy cross-check:")
        log.info(f"  Total generation (import): {total_import:>10.1f} GWh")
        log.info(f"  Total demand (export):     {total_export:>10.1f} GWh")
        log.info(f"  Difference (losses):       {losses:>10.1f} GWh "
                 f"({losses_pct:.1f}%)")
        log.info(f"{'='*60}")

    log.info("Done.")


if __name__ == "__main__":
    main()
