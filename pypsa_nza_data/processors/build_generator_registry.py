#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_generator_registry.py

Pre-process the EA dispatched generation plant file into a clean generator
registry for use by pypsa-nza-net.

Inputs
------
data/raw/static/20250917_DispatchedGenerationPlant.csv
    Raw EA file.  One row per (plant, operator, effective period) combination.
    The same physical unit appears multiple times when operators share capacity
    or when the operator changed over time, and some plants are registered under
    more than one PointOfConnectionCode.

data/raw/static/generator_overrides.csv  (optional)
    Manual corrections for known data quality issues in the EA source file.
    Columns: site, volts_kv, carrier, p_nom_mw_override, note
    Any row matching (site, volts_kv, carrier) has its p_nom_mw replaced
    with p_nom_mw_override.  The note column is logged for traceability.
    If the file does not exist the script continues without overrides.

Outputs
-------
data/processed/static/generator_registry.csv
    One row per (site, voltage, carrier) combination.
    Suitable for direct ingestion by nza_base_net._add_generators().

Processing steps
----------------
1.  Rename columns to clean snake_case names.
2.  Filter to currently active plants only.
3.  Parse PointOfConnectionCode into site, voltage tier, and sequence fields.
4.  Exclude sub-transmission voltages (below 110 kV).
5.  Map TechnologyCode + FuelCode to a standardised carrier name.
6.  Stage-1 deduplication: same physical unit registered under multiple
    operators or multiple POC codes.
    Group by (unit_code, tech_code, volts_kv) -> max(p_nom_mw).
7.  Stage-2 aggregation: different units at the same bus.
    Group by (site, volts_kv, carrier) -> sum(p_nom_mw).
8.  Apply manual overrides to correct known EA data quality issues.
9.  Assign bus names: sites with generators at more than one voltage level
    get suffixed names (e.g. ROX_220, ROX_110).
10. Write output.

Author: Phillippe Bruneau
"""

import sys
import logging
from pathlib import Path

import pandas as pd


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Raw column name -> clean name
COLUMN_MAP = {
    "PlantName":             "plant_name",
    "PointOfConnectionCode": "poc",
    "UnitCode":              "unit_code",
    "TechnologyCode":        "tech_code",
    "Technology":            "technology",
    "FuelCode":              "fuel_code",
    "Fuel":                  "fuel",
    "PeakingPlantFlag":      "peaking",
    "PlantOperatorCode":     "operator_code",
    "PlantOperator":         "operator",
    "DateCommissioned":      "date_commissioned",
    "DateDecommissioned":    "date_decommissioned",
    "NameplateMegawatts":    "p_nom_mw",
    "EffectiveStartDate":    "effective_start",
    "EffectiveEndDate":      "effective_end",
}

# Voltage tier encoded in POC characters 3-4 (0-indexed)
VOLT_TIER_MAP = {
    "22": 220,
    "11": 110,
    "06": 66,
    "03": 33,
    "01": 11,
}

# Minimum voltage (kV) to include in the transmission network
VOLT_THRESHOLD_KV = 110

# (TechnologyCode, FuelCode) -> carrier name
# fuel_code='*' acts as a wildcard matched after exact lookup fails
CARRIER_MAP = {
    ("HYD",  "*"):   "hydro",
    ("ONW",  "*"):   "wind",
    ("PV",   "*"):   "solar_pv",
    ("GEO",  "*"):   "geothermal",
    ("CCGT", "*"):   "ccgt",
    ("OCGT", "GAS"): "ocgt",
    ("OCGT", "DSL"): "diesel",
    ("BAT",  "*"):   "battery",
    ("COG",  "*"):   "gas",
    ("RKN",  "CLG"): "coal_gas",
    ("RKN",  "*"):   "coal_gas",
}

# TechnologyCode values that represent placeholders or decommissioned stubs
EXCLUDE_TECH_CODES = {"N/A"}

# Output column order
OUTPUT_COLUMNS = [
    "plant_name",
    "site",
    "poc",
    "volts_kv",
    "bus",
    "carrier",
    "p_nom_mw",
    "build_year",
    "peaking",
]


# =============================================================================
# CARRIER MAPPING
# =============================================================================

def map_carrier(tech_code: str, fuel_code: str) -> str:
    """
    Map (TechnologyCode, FuelCode) to a standardised carrier name.

    Tries exact (tech, fuel) match first, then (tech, '*') wildcard.

    Parameters
    ----------
    tech_code : str
    fuel_code : str

    Returns
    -------
    str
        Carrier name, or 'unknown' if no mapping exists.
    """
    key_exact = (tech_code.strip(), fuel_code.strip())
    key_wild  = (tech_code.strip(), "*")

    if key_exact in CARRIER_MAP:
        return CARRIER_MAP[key_exact]
    if key_wild in CARRIER_MAP:
        return CARRIER_MAP[key_wild]

    log.warning("No carrier mapping for tech=%s fuel=%s -> 'unknown'",
                tech_code, fuel_code)
    return "unknown"


# =============================================================================
# POC PARSING
# =============================================================================

def parse_poc(poc: str) -> tuple:
    """
    Parse a PointOfConnectionCode into (site, volts_kv, seq).

    POC format: SSSVVnn
        SSS - site code  (chars 0-2, upper case)
        VV  - voltage tier (chars 3-4: 22=220kV, 11=110kV, 06=66kV, 03=33kV)
        nn  - sequence number (chars 5-6)

    Parameters
    ----------
    poc : str
        Raw POC code, e.g. 'ROX2201', 'HLY2201', 'ARI1101'.

    Returns
    -------
    tuple : (site, volts_kv, seq)
        site     - 3-char site code
        volts_kv - integer kV, or None if voltage tier unrecognised
        seq      - 2-char sequence string
    """
    poc = str(poc).strip()
    if len(poc) < 7:
        log.warning("Unexpected POC length: '%s'", poc)
        return poc[:3].upper(), None, ""

    site     = poc[:3].upper()
    tier_str = poc[3:5]
    seq      = poc[5:7]
    volts_kv = VOLT_TIER_MAP.get(tier_str)

    if volts_kv is None:
        log.warning("Unrecognised voltage tier '%s' in POC '%s'", tier_str, poc)

    return site, volts_kv, seq


# =============================================================================
# MANUAL OVERRIDES
# =============================================================================

def load_overrides(overrides_path: Path) -> pd.DataFrame:
    """
    Load the manual overrides file.

    Parameters
    ----------
    overrides_path : Path
        Path to generator_overrides.csv.

    Returns
    -------
    pd.DataFrame
        Overrides table, or empty DataFrame if file absent or unreadable.
    """
    if not overrides_path.exists():
        log.warning("Overrides file not found: %s", overrides_path)
        log.warning("  Continuing without manual overrides.")
        return pd.DataFrame(
            columns=["site", "volts_kv", "carrier", "p_nom_mw_override", "note"]
        )

    try:
        df_ov = pd.read_csv(overrides_path, dtype=str)
        df_ov["volts_kv"]         = df_ov["volts_kv"].astype(int)
        df_ov["p_nom_mw_override"] = pd.to_numeric(
            df_ov["p_nom_mw_override"], errors="coerce"
        )
        log.info("  Loaded %d override(s) from %s", len(df_ov), overrides_path)
        return df_ov
    except Exception as exc:
        log.error("Failed to load overrides file: %s", exc)
        return pd.DataFrame(
            columns=["site", "volts_kv", "carrier", "p_nom_mw_override", "note"]
        )


def apply_overrides(df: pd.DataFrame, df_ov: pd.DataFrame) -> pd.DataFrame:
    """
    Apply manual capacity overrides to the aggregated registry.

    Each override row matches on (site, volts_kv, carrier) and replaces
    p_nom_mw with p_nom_mw_override.  Unmatched overrides are reported
    as warnings so stale entries in the overrides file are visible.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated generator registry.
    df_ov : pd.DataFrame
        Overrides table from load_overrides().

    Returns
    -------
    pd.DataFrame
        Registry with overrides applied.
    """
    if df_ov.empty:
        return df

    for _, ov in df_ov.iterrows():
        mask = (
            (df["site"]     == ov["site"]) &
            (df["volts_kv"] == int(ov["volts_kv"])) &
            (df["carrier"]  == ov["carrier"])
        )
        matches = mask.sum()

        if matches == 0:
            log.warning(
                "Override has no matching registry row: site=%s volts=%s "
                "carrier=%s -- override ignored",
                ov["site"], ov["volts_kv"], ov["carrier"]
            )
            continue

        old_val = df.loc[mask, "p_nom_mw"].values[0]
        df.loc[mask, "p_nom_mw"] = ov["p_nom_mw_override"]
        log.info(
            "  Override applied: %s %s kV %s  %.1f MW -> %.1f MW  (%s)",
            ov["site"], ov["volts_kv"], ov["carrier"],
            old_val, ov["p_nom_mw_override"],
            ov.get("note", "")
        )

    return df


# =============================================================================
# BUS NAME ASSIGNMENT
# =============================================================================

def assign_bus_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign PyPSA bus names.

    Sites with generators at more than one voltage level receive suffixed
    bus names (e.g. ROX_220, ROX_110).  Single-voltage sites keep their
    plain 3-character site code.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'site' and 'volts_kv'.

    Returns
    -------
    pd.DataFrame
        Input with new 'bus' column added.
    """
    volts_per_site   = df.groupby("site")["volts_kv"].nunique()
    multi_volt_sites = set(volts_per_site[volts_per_site > 1].index)

    def _name(row):
        if row["site"] in multi_volt_sites:
            return f"{row['site']}_{int(row['volts_kv'])}"
        return row["site"]

    df["bus"] = df.apply(_name, axis=1)
    return df


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def build_generator_registry(
    input_path: Path,
    output_path: Path,
    overrides_path: Path,
) -> pd.DataFrame:
    """
    Build the generator registry from the raw EA dispatched generation file.

    Parameters
    ----------
    input_path : Path
        Raw EA CSV file.
    output_path : Path
        Destination for the processed registry CSV.
    overrides_path : Path
        Manual overrides CSV (may be absent).

    Returns
    -------
    pd.DataFrame
        Processed generator registry.
    """
    log.info("=" * 60)
    log.info("BUILD GENERATOR REGISTRY")
    log.info("=" * 60)
    log.info("Input    : %s", input_path)
    log.info("Output   : %s", output_path)
    log.info("Overrides: %s", overrides_path)

    # ------------------------------------------------------------------
    # STEP 1: Load and rename columns
    # ------------------------------------------------------------------
    log.info("\nStep 1: Loading raw data")
    df = pd.read_csv(input_path, dtype=str)
    log.info("  Loaded %d rows, %d columns", len(df), len(df.columns))

    missing_cols = set(COLUMN_MAP.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing expected columns in input file: {sorted(missing_cols)}"
        )

    df = df.rename(columns=COLUMN_MAP)
    df["p_nom_mw"] = pd.to_numeric(df["p_nom_mw"], errors="coerce")

    # ------------------------------------------------------------------
    # STEP 2: Filter active plants
    # ------------------------------------------------------------------
    log.info("\nStep 2: Filtering active plants")
    n_before = len(df)

    # effective_end == '9999-12-31' flags the currently valid registration row
    df = df[df["effective_end"] == "9999-12-31"].copy()
    log.info("  After effective_end filter : %d rows (removed %d)",
             len(df), n_before - len(df))

    n_before = len(df)
    # date_decommissioned is NaN (not the string "null") for active plants.
    # pandas parses the source string "null" as NaN when reading with dtype=str
    # in some versions.  isna() handles both cases correctly.
    df = df[df["date_decommissioned"].isna()].copy()
    log.info("  After decommission filter  : %d rows (removed %d)",
             len(df), n_before - len(df))

    # ------------------------------------------------------------------
    # STEP 3: Parse POC codes
    # ------------------------------------------------------------------
    log.info("\nStep 3: Parsing POC codes")
    parsed       = df["poc"].apply(parse_poc)
    df["site"]   = parsed.apply(lambda t: t[0])
    df["volts_kv"] = parsed.apply(lambda t: t[1])

    # ------------------------------------------------------------------
    # STEP 4: Apply voltage threshold
    # ------------------------------------------------------------------
    log.info("\nStep 4: Applying voltage threshold (>= %d kV)", VOLT_THRESHOLD_KV)
    n_before = len(df)

    df = df[df["volts_kv"].notna()].copy()
    df["volts_kv"] = df["volts_kv"].astype(int)
    df = df[df["volts_kv"] >= VOLT_THRESHOLD_KV].copy()

    log.info("  After voltage filter: %d rows (removed %d)",
             len(df), n_before - len(df))

    # ------------------------------------------------------------------
    # STEP 5: Exclude placeholder tech codes
    # ------------------------------------------------------------------
    log.info("\nStep 5: Excluding tech codes %s", EXCLUDE_TECH_CODES)
    n_before = len(df)
    df = df[~df["tech_code"].isin(EXCLUDE_TECH_CODES)].copy()
    log.info("  After tech_code filter: %d rows (removed %d)",
             len(df), n_before - len(df))

    # ------------------------------------------------------------------
    # STEP 6: Map carriers
    # ------------------------------------------------------------------
    log.info("\nStep 6: Mapping technology codes to carriers")
    df["carrier"] = df.apply(
        lambda row: map_carrier(row["tech_code"], row["fuel_code"]),
        axis=1,
    )

    unknown_mask = df["carrier"] == "unknown"
    if unknown_mask.any():
        log.warning("  %d rows with unknown carrier (will be excluded):",
                    unknown_mask.sum())
        for _, row in df[unknown_mask].iterrows():
            log.warning("    poc=%s tech=%s fuel=%s  %s",
                        row["poc"], row["tech_code"],
                        row["fuel_code"], row["plant_name"])
        df = df[~unknown_mask].copy()

    # ------------------------------------------------------------------
    # STEP 7: Extract build year
    # ------------------------------------------------------------------
    log.info("\nStep 7: Extracting build year")
    df["build_year"] = (
        pd.to_datetime(df["date_commissioned"], errors="coerce")
        .dt.year.fillna(0).astype(int)
    )

    # ------------------------------------------------------------------
    # STEP 8: Stage-1 deduplication
    #
    # The EA file registers some physical units under more than one POC code
    # (e.g. Fonterra/Whareroa appears under both HWA1101 and HWA1102).  It also
    # records the same unit multiple times when the operator changed.  Grouping
    # by (unit_code, tech_code, volts_kv) collapses all registrations of the
    # same physical machine at the same voltage into a single row, while keeping
    # units that genuinely connect at different voltages separate (e.g. ROX0
    # appears at both 110 kV and 220 kV at Roxburgh and must remain two rows).
    # max(p_nom_mw) retains the highest reported nameplate rating.
    # ------------------------------------------------------------------
    log.info("\nStep 8: Stage-1 deduplication - same unit, multiple operators/POCs")
    log.info("  Group by (unit_code, tech_code, volts_kv) -> max(p_nom_mw)")
    n_before = len(df)

    df = (
        df.groupby(["unit_code", "tech_code", "volts_kv"], as_index=False)
        .agg(
            plant_name=("plant_name", "first"),
            poc=("poc",        "first"),
            site=("site",       "first"),
            carrier=("carrier",   "first"),
            fuel_code=("fuel_code",  "first"),
            p_nom_mw=("p_nom_mw",   "max"),
            build_year=("build_year", "min"),
            peaking=("peaking",    "first"),
        )
    )
    log.info("  Reduced from %d to %d rows (removed %d duplicates)",
             n_before, len(df), n_before - len(df))

    # ------------------------------------------------------------------
    # STEP 9: Stage-2 aggregation
    #
    # Different physical units of the same technology connecting at the same
    # site and voltage are summed to give the total installed capacity at
    # that bus (e.g. Huntly Rankine units 1, 2, 4 -> 750 MW coal_gas at HLY).
    # ------------------------------------------------------------------
    log.info("\nStep 9: Stage-2 aggregation - sum units at same bus")
    log.info("  Group by (site, volts_kv, carrier) -> sum(p_nom_mw)")
    n_before = len(df)

    df = (
        df.groupby(["site", "volts_kv", "carrier"], as_index=False)
        .agg(
            plant_name=("plant_name", "first"),
            poc=("poc",        "first"),
            p_nom_mw=("p_nom_mw",   "sum"),
            build_year=("build_year", "min"),
            peaking=("peaking",    "first"),
        )
    )
    log.info("  Reduced from %d to %d rows (aggregated %d unit groups)",
             n_before, len(df), n_before - len(df))

    # ------------------------------------------------------------------
    # STEP 10: Apply manual overrides
    # ------------------------------------------------------------------
    log.info("\nStep 10: Applying manual overrides")
    df_ov = load_overrides(overrides_path)
    df    = apply_overrides(df, df_ov)

    # ------------------------------------------------------------------
    # STEP 11: Assign bus names
    # ------------------------------------------------------------------
    log.info("\nStep 11: Assigning bus names")
    df = assign_bus_names(df)

    multi_volt = df[df["bus"].str.contains("_")]["site"].unique()
    log.info("  Multi-voltage sites (%d): %s",
             len(multi_volt), sorted(multi_volt))

    # ------------------------------------------------------------------
    # STEP 12: Sort and write output
    # ------------------------------------------------------------------
    log.info("\nStep 12: Writing output")
    df = (
        df[OUTPUT_COLUMNS]
        .sort_values(["site", "volts_kv", "carrier"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("  Written %d generators to %s", len(df), output_path)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    log.info("\n%s", "=" * 60)
    log.info("REGISTRY SUMMARY")
    log.info("%s", "=" * 60)
    log.info("  Total generator entries : %d", len(df))
    log.info("  Unique sites            : %d", df["site"].nunique())
    log.info("  Unique buses            : %d", df["bus"].nunique())
    log.info("\n  Capacity by carrier (MW):")

    cs = (
        df.groupby("carrier")["p_nom_mw"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total_mw", "count": "n_entries"})
        .sort_values("total_mw", ascending=False)
    )
    for carrier, row in cs.iterrows():
        log.info("    %-15s %7.1f MW  (%d entries)",
                 carrier, row["total_mw"], row["n_entries"])

    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """
    Run the generator registry build with default workspace paths.

    Directory layout assumed:
        pypsa_nza_workspace/
            data/
                raw/static/      <-- input files live here
                processed/static/<-- output written here
            pypsa-nza-data/
                build_generator_registry.py  <-- this script
    """
    import os
    _root = os.environ.get("PYPSA_NZA_ROOT", "")
    workspace = Path(_root) / "pypsa_nza_workspace"
    
    input_path     = (workspace / "data" / "raw" / "static"
                      / "20250917_DispatchedGenerationPlant.csv")
    overrides_path = (workspace / "data" / "raw" / "static"
                      / "generator_overrides.csv")
    output_path    = (workspace / "data" / "processed" / "static"
                      / "generator_registry.csv")

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    build_generator_registry(input_path, output_path, overrides_path)
    log.info("\nDone.")


if __name__ == "__main__":
    main()

line	name	description	bus0	bus1	ss0	ss1	volts	cofing	length	distance	Unnamed: 11	Unnamed: 12	delta	Unnamed: 14

