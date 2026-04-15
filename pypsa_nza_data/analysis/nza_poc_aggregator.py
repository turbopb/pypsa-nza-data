# -*- coding: utf-8 -*-

"""
poc_aggregator.py

Reads a single Transpower grid_export or grid_import CSV file and aggregates
half-hourly POC time series to site level, with optional voltage threshold filtering.

Usage:
    python poc_aggregator.py --file grid_export_2024_07.csv --threshold 110
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree



# --- Logging setup ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# --- POC parsing ------------------------------------------------------------

def parse_poc_code(poc: str) -> tuple[str, int, int]:
    """
    Parse a POC code into its components.

    Parameters
    ----------
    poc : str
        POC code, e.g. 'ALB0331' or 'ARA2201'.

    Returns
    -------
    site : str
        Three-letter site code, e.g. 'ALB'.
    voltage : int
        Nominal voltage in kV, e.g. 33 or 220.
    busbar : int
        Busbar index, e.g. 1 or 2.

    Raises
    ------
    ValueError
        If the POC code does not conform to expected format.
    """
    if len(poc) != 7:
        raise ValueError(f"Unexpected POC code length: '{poc}' "
                         f"(expected 7 characters)")
    site    = poc[0:3].upper()
    voltage = int(poc[3:6])
    busbar  = int(poc[6])
    return site, voltage, busbar


def get_site(poc: str) -> str:
    """Return site code from POC string."""
    return parse_poc_code(poc)[0]


def get_voltage(poc: str) -> int:
    """Return voltage level in kV from POC string."""
    return parse_poc_code(poc)[1]


# --- File loading -----------------------------------------------------------

def load_poc_file(filepath: Path) -> pd.DataFrame:
    """
    Load a Transpower grid_export or grid_import CSV file.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and POC codes as columns.
        Values in MW (half-hourly).
    """
    log.info(f"Loading file: {filepath}")

    df = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True
    )

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("First column could not be parsed as datetime. "
                         "Check file format.")

    # Strip any whitespace from column names
    df.columns = df.columns.str.strip()

    log.info(f"  Loaded {len(df)} timesteps x {len(df.columns)} POCs")
    log.info(f"  Period: {df.index[0]} to {df.index[-1]}")
    n_days = (df.index[-1].date() - df.index[0].date()).days + 1
    
    n_days = (df.index[-1].date() - df.index[0].date()).days + 1
    log.info(f"  Expected timesteps for period: "
         f"{n_days * 48} ({n_days} days x 48 half-hours)")

    return df


# --- POC inventory ----------------------------------------------------------

def build_poc_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table of all POCs present in the file.

    Parameters
    ----------
    df : pd.DataFrame
        Raw POC DataFrame.

    Returns
    -------
    inventory : pd.DataFrame
        Table with columns: poc, site, voltage, busbar, 
        n_nonzero, total_energy.
    """
    records = []
    for poc in df.columns:
        try:
            site, voltage, busbar = parse_poc_code(poc)
        except ValueError as e:
            log.warning(f"Skipping malformed POC '{poc}': {e}")
            continue

        records.append({
            "poc"         : poc,
            "site"        : site,
            "voltage_kv"  : voltage,
            "busbar"      : busbar,
            "n_nonzero"   : (df[poc] != 0).sum(),
            "total_energy": df[poc].sum()
        })

    inventory = pd.DataFrame(records).set_index("poc")
    return inventory


# --- Voltage filtering ------------------------------------------------------

def filter_by_voltage(df: pd.DataFrame, 
                       threshold_kv: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split POC columns into those at or above threshold and those below.

    Parameters
    ----------
    df : pd.DataFrame
        Raw POC DataFrame.
    threshold_kv : int
        Minimum voltage to include explicitly (e.g. 110).

    Returns
    -------
    df_above : pd.DataFrame
        POCs at or above threshold_kv.
    df_below : pd.DataFrame
        POCs below threshold_kv.
    """
    above_cols = [c for c in df.columns if get_voltage(c) >= threshold_kv]
    below_cols = [c for c in df.columns if get_voltage(c) <  threshold_kv]

    log.info(f"  POCs at or above {threshold_kv} kV: {len(above_cols)}")
    log.info(f"  POCs below {threshold_kv} kV:       {len(below_cols)}")

    return df[above_cols], df[below_cols]


# --- Site aggregation -------------------------------------------------------

def aggregate_site_demand(df: pd.DataFrame, 
                           site_code: str) -> pd.Series:
    """
    Sum all POC columns belonging to a given site.

    Parameters
    ----------
    df : pd.DataFrame
        POC DataFrame (may be pre-filtered by voltage).
    site_code : str
        Three-letter site code, e.g. 'ISL'.

    Returns
    -------
    pd.Series
        Aggregated time series for the site.
    """
    site_cols = [c for c in df.columns if get_site(c) == site_code.upper()]
    if not site_cols:
        log.warning(f"No POCs found for site '{site_code}'")
        return pd.Series(0.0, index=df.index, name=site_code)
    return df[site_cols].sum(axis=1).rename(site_code)


def aggregate_all_sites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all POC columns to site level.

    Parameters
    ----------
    df : pd.DataFrame
        POC DataFrame.

    Returns
    -------
    pd.DataFrame
        Site-aggregated DataFrame with site codes as columns.
    """
    sites = sorted({get_site(c) for c in df.columns})
    log.info(f"  Aggregating {len(df.columns)} POCs to {len(sites)} sites")

    site_series = [aggregate_site_demand(df, s) for s in sites]
    return pd.concat(site_series, axis=1)


# --- Energy accounting ------------------------------------------------------

def energy_check(df_raw: pd.DataFrame, 
                 df_aggregated: pd.DataFrame,
                 label: str = "") -> None:
    """
    Verify that aggregation preserves total energy.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Original POC DataFrame.
    df_aggregated : pd.DataFrame
        Site-aggregated DataFrame.
    label : str
        Label for logging output.
    """
    total_raw  = df_raw.sum().sum()
    total_agg  = df_aggregated.sum().sum()
    delta      = abs(total_raw - total_agg)
    pct        = delta / total_raw * 100 if total_raw != 0 else 0

    log.info(f"  Energy check {label}:")
    log.info(f"    Raw total:        {total_raw:>12.1f} MWh")
    log.info(f"    Aggregated total: {total_agg:>12.1f} MWh")
    log.info(f"    Difference:       {delta:>12.4f} MWh ({pct:.4f}%)")

    if pct > 0.01:
        log.warning(" Energy check FAILED - aggregation does not preserve total")
    else:
        log.info("  Energy check PASSED")


def sub_threshold_fraction(df_raw: pd.DataFrame, 
                            threshold_kv: int) -> None:
    """
    Report what fraction of total energy is at sub-threshold voltages.
    """
    total = df_raw.sum().sum()
    sub_cols = [c for c in df_raw.columns if get_voltage(c) < threshold_kv]
    sub_total = df_raw[sub_cols].sum().sum() if sub_cols else 0.0
    pct = sub_total / total * 100 if total != 0 else 0

    log.info(f"  Sub-threshold (<{threshold_kv} kV) energy fraction: "
             f"{sub_total:.1f} / {total:.1f} MWh = {pct:.2f}%")

    if pct > 5.0:
        log.warning(f"Sub-threshold fraction exceeds 5% "
                    f"consider geographic aggregation rather than exclusion")

def load_sites(nodes_csv: Path, sites_csv: Path = None) -> pd.DataFrame:
    """
    Load site coordinates from nodes.csv, optionally supplemented
    by Sites.csv for any missing entries.

    Parameters
    ----------
    nodes_csv : Path
        Path to nodes.csv (primary source).
    sites_csv : Path, optional
        Path to Sites.csv (fallback for missing sites).

    Returns
    -------
    pd.DataFrame
        Index = site code, columns = X, Y, long, lat, name, 
        zone, island, volts
    """
    df = pd.read_csv(nodes_csv, sep=None, engine="python")
    df = df.set_index("site")
    log.info(f"Loaded {len(df)} sites from {nodes_csv.name}")

    if sites_csv is not None and sites_csv.exists():
        df_sites = pd.read_csv(sites_csv)
        df_sites = df_sites[df_sites["status"] == "COMMISSIONED"].copy()
        df_sites = df_sites.set_index("MXLOCATION")
        df_sites = df_sites.rename(columns={"description": "name"})

        # Find sites in Sites.csv but not in nodes.csv
        missing = df_sites.index.difference(df.index)
        if len(missing) > 0:
            log.info(f"Adding {len(missing)} sites from Sites.csv "
                     f"not found in nodes.csv: {missing.tolist()}")
            df_extra = df_sites.loc[missing, ["X", "Y", "name"]].copy()
            df_extra["long"] = None
            df_extra["lat"]  = None
            df = pd.concat([df, df_extra])

    return df

def build_nearest_bus_mapping(
        sub_threshold_sites: list[str],
        modelled_sites: list[str],
        sites_df: pd.DataFrame) -> pd.Series:
    """
    For each sub-threshold site, find the nearest modelled bus
    using NZTM coordinates.

    Parameters
    ----------
    sub_threshold_sites : list of str
        Site codes below voltage threshold.
    modelled_sites : list of str
        Site codes at or above voltage threshold.
    sites_df : pd.DataFrame
        Sites dataframe with X, Y columns, indexed by site code.

    Returns
    -------
    pd.Series
        Index = sub-threshold site code, values = nearest modelled site code.
    """
    # Filter to sites that exist in Sites.csv
    modelled_known   = [s for s in modelled_sites   if s in sites_df.index]
    sub_thresh_known = [s for s in sub_threshold_sites if s in sites_df.index]

    missing_mod = set(modelled_sites) - set(modelled_known)
    missing_sub = set(sub_threshold_sites) - set(sub_thresh_known)

    if missing_mod:
        log.warning(f"  Modelled sites not found in Sites.csv: {missing_mod}")
    if missing_sub:
        log.warning(f"  Sub-threshold sites not found in Sites.csv: {missing_sub}")

    # Build KD-tree from modelled site coordinates
    modelled_coords = sites_df.loc[modelled_known, ["X", "Y"]].values
    sub_coords      = sites_df.loc[sub_thresh_known, ["X", "Y"]].values

    tree = cKDTree(modelled_coords)
    distances, indices = tree.query(sub_coords)

    mapping = pd.Series(
        index=sub_thresh_known,
        data=[modelled_known[i] for i in indices],
        name="nearest_modelled_bus"
    )

    # Log distance summary
    distances_km = distances / 1000.0
    log.info(f"  Nearest bus mapping summary:")
    log.info(f"    Mean distance:   {distances_km.mean():.1f} km")
    log.info(f"    Max distance:    {distances_km.max():.1f} km")
    log.info(f"    Sites > 50 km:   "
             f"{(distances_km > 50).sum()} of {len(distances_km)}")

    # Flag any sites very far from nearest modelled bus
    far_sites = [sub_thresh_known[i] for i, d in enumerate(distances_km) 
                 if d > 50]
    if far_sites:
        log.warning(f"  Sites more than 50 km from nearest modelled bus: "
                    f"{far_sites}")

    return mapping


def apply_geographic_aggregation(
        df_above: pd.DataFrame,
        df_below: pd.DataFrame,
        sites_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sub-threshold site demand to nearest modelled bus
    and add to above-threshold site aggregation.

    Parameters
    ----------
    df_above : pd.DataFrame
        Site-aggregated demand at or above voltage threshold.
    df_below : pd.DataFrame
        Site-aggregated demand below voltage threshold.
    sites_df : pd.DataFrame
        Sites dataframe.

    Returns
    -------
    pd.DataFrame
        df_above with sub-threshold demand added to nearest modelled bus.
    """
    modelled_sites     = df_above.columns.tolist()
    sub_thresh_sites   = df_below.columns.tolist()

    log.info(f"Applying geographic aggregation of "
             f"{len(sub_thresh_sites)} sub-threshold sites "
             f"to {len(modelled_sites)} modelled buses...")

    mapping = build_nearest_bus_mapping(
        sub_thresh_sites, modelled_sites, sites_df
    )

    df_result = df_above.copy()

    for sub_site, nearest_bus in mapping.items():
        if sub_site in df_below.columns and nearest_bus in df_result.columns:
            df_result[nearest_bus] += df_below[sub_site]
        else:
            log.warning(f"  Could not map {sub_site} -> {nearest_bus}")

    # Energy check
    total_before = df_above.sum().sum() + df_below.sum().sum()
    total_after  = df_result.sum().sum()
    delta_pct    = abs(total_before - total_after) / total_before * 100
    log.info(f"  Geographic aggregation energy check:")
    log.info(f"    Before: {total_before:.1f} MWh")
    log.info(f"    After:  {total_after:.1f} MWh")
    log.info(f"    Delta:  {delta_pct:.4f}%")
    if delta_pct > 0.01:
        log.warning("  Geographic aggregation energy check FAILED")
    else:
        log.info("  Geographic aggregation energy check PASSED")

    return df_result

def aggregate_by_site_all_voltages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ALL POC columns to site level regardless of voltage.
    This is the correct approach for transmission network modelling --
    energy must flow through the high-voltage network to reach any
    distribution substation, so all demand is correctly attributed
    to the site bus regardless of exit voltage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw POC DataFrame with all voltage levels.

    Returns
    -------
    pd.DataFrame
        Site-aggregated DataFrame.
    """
    sites = sorted({get_site(c) for c in df.columns})
    log.info(f"Aggregating {len(df.columns)} POCs across all voltages "
             f"to {len(sites)} sites")
    site_series = [aggregate_site_demand(df, s) for s in sites]
    return pd.concat(site_series, axis=1)

    
# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Transpower POC export/import file to site level."
    )
    parser.add_argument("--file",      required=True,
                        help="Path to input CSV file")
    parser.add_argument("--threshold", type=int, default=110,
                        help="Minimum voltage kV for reporting (default: 110)")
    parser.add_argument("--outdir",    default=".",
                        help="Output directory (default: current directory)")
    args = parser.parse_args()

    filepath = Path(args.file)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load ---------------------------------------------------------------
    df_raw = load_poc_file(filepath)

    # --- Inventory ----------------------------------------------------------
    log.info("Building POC inventory...")
    inventory = build_poc_inventory(df_raw)
    inv_path  = outdir / f"{filepath.stem}_inventory.csv"
    inventory.to_csv(inv_path)
    log.info(f"  Inventory written to: {inv_path}")

    # Voltage level summary
    log.info("Voltage level breakdown:")
    for v, grp in inventory.groupby("voltage_kv"):
        log.info(f"  {v:>4} kV : {len(grp):>4} POCs, "
                 f"total energy = {grp['total_energy'].sum():>10.1f} MWh")

    # --- Sub-threshold fraction (informational only) -----------------------
    log.info("Checking sub-threshold energy fraction...")
    sub_threshold_fraction(df_raw, args.threshold)

    # --- Aggregate all POCs to site level regardless of voltage ------------
    log.info("Aggregating all POCs to site level...")
    df_sites_all = aggregate_by_site_all_voltages(df_raw)
    energy_check(df_raw, df_sites_all, label="(all voltages to sites)")

    all_path = outdir / f"{filepath.stem}_sites_all.csv"
    df_sites_all.to_csv(all_path)
    log.info(f"  Full site aggregation written to: {all_path}")

#     # --- Identify sites with no high-voltage connection --------------------
#     # populate modelled_network_sites from line CSV later
#     modelled_network_sites = set()
#     poc_sites    = set(df_sites_all.columns)
#     unconnected  = poc_sites - modelled_network_sites
#     if unconnected:
#         log.info(f"Sites in POC data with no network connection: "
#                  f"{sorted(unconnected)}")
#         log.info(f"These will need geographic mapping to nearest network bus")

    # --- Summary ------------------------------------------------------------
    log.info("=== Summary ===")
    log.info(f"  Input file:   {filepath.name}")
    log.info(f"  Timesteps:    {len(df_raw)}")
    log.info(f"  Total POCs:   {len(df_raw.columns)}")
    log.info(f"  Total sites:  {len(df_sites_all.columns)}")
    log.info("Done.")


if __name__ == "__main__":
    main()

    
