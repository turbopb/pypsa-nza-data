# -*- coding: utf-8 -*-
"""
plot_annual_overview.py

Plots annual overview of generation dispatch, grid import and grid export
time series for a given year. Includes validation plots for the discrepancy
between generation dispatch and grid import totals.

Usage:
    python plot_annual_overview.py --year 2024 --anndir ./annual --outdir ./plots

    # Plot only three-dataset overview
    python plot_annual_overview.py --year 2024 --no-import-export --no-validation

    # Plot only import/export (model inputs)
    python plot_annual_overview.py --year 2024 --no-all-three --no-validation

    # Plot only validation
    python plot_annual_overview.py --year 2024 --no-all-three --no-import-export
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Colour scheme ----------------------------------------------------------
# Consistent across all plots in this script
COL_GEN = "steelblue"
COL_IMP = "darkorange"
COL_EXP = "seagreen"


# ============================================================================
# Data loading
# ============================================================================

def load_annual(anndir: Path, year: int, flow: str) -> pd.Series:
    """
    Load annual site-aggregated file and return total across all sites.

    Parameters
    ----------
    anndir : Path
        Directory containing annual CSV files.
    year : int
        Year to load.
    flow : str
        'export', 'import', or 'gen'.

    Returns
    -------
    pd.Series
        Half-hourly total across all sites, DatetimeIndex.
    """
    path = anndir / f"{year}_{flow}_sites_all.csv"
    log.info(f"Loading {path.name}...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sum(axis=1).rename(flow)


def load_annual_df(anndir: Path, year: int, flow: str) -> pd.DataFrame:
    """
    Load annual site-aggregated file and return full DataFrame (one column per site).

    Parameters
    ----------
    anndir : Path
        Directory containing annual CSV files.
    year : int
        Year to load.
    flow : str
        'export', 'import', or 'gen'.

    Returns
    -------
    pd.DataFrame
        Half-hourly site-level DataFrame.
    """
    path = anndir / f"{year}_{flow}_sites_all.csv"
    log.info(f"Loading site-level {path.name}...")
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ============================================================================
# Group A -- All three datasets
# ============================================================================

def plot_annual_timeseries(gen: pd.Series,
                            imp: pd.Series,
                            exp: pd.Series,
                            year: int,
                            outdir: Path) -> None:
    """Plot full year daily totals for all three datasets."""

    gen_d = gen.resample("D").sum() / 1000.0
    imp_d = imp.resample("D").sum() / 1000.0
    exp_d = exp.resample("D").sum() / 1000.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gen_d.index, gen_d.values,
            label="Generation dispatch", color=COL_GEN, linewidth=1.0)
    ax.plot(imp_d.index, imp_d.values,
            label="Grid import",         color=COL_IMP, linewidth=1.0)
    ax.plot(exp_d.index, exp_d.values,
            label="Grid export (demand)", color=COL_EXP, linewidth=1.0)

    ax.set_title(f"Daily energy totals -- New Zealand National Grid {year}",
                 fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (GWh/day)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = outdir / f"{year}_annual_timeseries.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_monthly_bar(gen: pd.Series,
                     imp: pd.Series,
                     exp: pd.Series,
                     year: int,
                     outdir: Path) -> None:
    """Plot monthly grouped bar chart for all three datasets."""

    gen_m  = gen.resample("ME").sum() / 1e6
    imp_m  = imp.resample("ME").sum() / 1e6
    exp_m  = exp.resample("ME").sum() / 1e6
    months = [d.strftime("%b") for d in gen_m.index]
    x      = range(len(months))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width for i in x], gen_m.values,
           width=width, label="Generation dispatch", color=COL_GEN)
    ax.bar([i           for i in x], imp_m.values,
           width=width, label="Grid import",         color=COL_IMP)
    ax.bar([i + width for i in x], exp_m.values,
           width=width, label="Grid export (demand)", color=COL_EXP)

    ax.set_title(f"Monthly energy totals -- New Zealand National Grid {year}",
                 fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Energy (TWh)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(months)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = outdir / f"{year}_monthly_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_weekly_detail(gen: pd.Series,
                        imp: pd.Series,
                        exp: pd.Series,
                        year: int,
                        outdir: Path,
                        week_start: str = None) -> None:
    """Plot one week of half-hourly data for all three datasets."""

    if week_start is None:
        week_start = f"{year}-07-07"

    start = pd.Timestamp(week_start)
    end   = start + pd.Timedelta(days=7)

    gen_w = gen[start:end] / 1000.0
    imp_w = imp[start:end] / 1000.0
    exp_w = exp[start:end] / 1000.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gen_w.index, gen_w.values,
            label="Generation dispatch", color=COL_GEN, linewidth=1.0)
    ax.plot(imp_w.index, imp_w.values,
            label="Grid import",         color=COL_IMP, linewidth=1.0)
    ax.plot(exp_w.index, exp_w.values,
            label="Grid export (demand)", color=COL_EXP, linewidth=1.0)

    ax.set_title(f"Half-hourly energy -- week of {week_start}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (GWh per half-hour)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()

    path = outdir / f"{year}_weekly_detail_{week_start}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


# ============================================================================
# Group B -- Import/export only (model inputs)
# ============================================================================

def plot_annual_timeseries_imp_exp(imp: pd.Series,
                                    exp: pd.Series,
                                    year: int,
                                    outdir: Path) -> None:
    """Plot full year daily totals for import and export (model inputs)."""

    imp_d = imp.resample("D").sum() / 1000.0
    exp_d = exp.resample("D").sum() / 1000.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(imp_d.index, imp_d.values,
            label="Grid import (generation)", color=COL_IMP, linewidth=1.0)
    ax.plot(exp_d.index, exp_d.values,
            label="Grid export (demand)",     color=COL_EXP, linewidth=1.0)

    ax.set_title(
        f"Daily energy totals -- PyPSA model inputs {year}",
        fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (GWh/day)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = outdir / f"{year}_annual_timeseries_imp_exp.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_monthly_bar_imp_exp(imp: pd.Series,
                              exp: pd.Series,
                              year: int,
                              outdir: Path) -> None:
    """Plot monthly grouped bar chart for import and export only."""

    imp_m  = imp.resample("ME").sum() / 1e6
    exp_m  = exp.resample("ME").sum() / 1e6
    months = [d.strftime("%b") for d in imp_m.index]
    x      = range(len(months))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width / 2 for i in x], imp_m.values,
           width=width, label="Grid import (generation)", color=COL_IMP)
    ax.bar([i + width / 2 for i in x], exp_m.values,
           width=width, label="Grid export (demand)",     color=COL_EXP)

    ax.set_title(
        f"Monthly energy totals -- PyPSA model inputs {year}",
        fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Energy (TWh)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(months)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = outdir / f"{year}_monthly_bar_imp_exp.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_weekly_detail_imp_exp(imp: pd.Series,
                                exp: pd.Series,
                                year: int,
                                outdir: Path,
                                week_start: str = None) -> None:
    """Plot one week of half-hourly import and export data."""

    if week_start is None:
        week_start = f"{year}-07-07"

    start = pd.Timestamp(week_start)
    end   = start + pd.Timedelta(days=7)

    imp_w = imp[start:end] / 1000.0
    exp_w = exp[start:end] / 1000.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(imp_w.index, imp_w.values,
            label="Grid import (generation)", color=COL_IMP, linewidth=1.0)
    ax.plot(exp_w.index, exp_w.values,
            label="Grid export (demand)",     color=COL_EXP, linewidth=1.0)

    ax.set_title(
        f"Half-hourly energy -- PyPSA model inputs -- week of {week_start}",
        fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (GWh per half-hour)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()

    path = outdir / f"{year}_weekly_detail_imp_exp_{week_start}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


# ============================================================================
# Group C -- Validation plots
# ============================================================================

def plot_validation_scatter(anndir: Path,
                             year: int,
                             outdir: Path) -> None:
    """
    Scatter plot of annual generation dispatch vs grid import at site level.

    Sites lying on the 1:1 diagonal are identical in both datasets.
    Deviations from the diagonal indicate discrepancies.
    """
    gen_df = load_annual_df(anndir, year, "gen")
    imp_df = load_annual_df(anndir, year, "import")

    # Annual totals per site
    gen_tot = gen_df.sum() / 1000.0
    imp_tot = imp_df.sum() / 1000.0

    # Common sites only
    common = gen_tot.index.intersection(imp_tot.index)
    gen_c  = gen_tot.loc[common]
    imp_c  = imp_tot.loc[common]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(gen_c.values, imp_c.values,
               color=COL_IMP, alpha=0.7, s=40, zorder=3)

    # 1:1 reference line
    lim_max = max(gen_c.max(), imp_c.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max],
            color="gray", linewidth=1.0, linestyle="--",
            label="1:1 line (identical)")

    # Label notable sites
    notable = ["MAN", "HLY", "BEN", "ROX", "KPO"]
    for site in notable:
        if site in common:
            ax.annotate(site,
                        xy=(gen_c[site], imp_c[site]),
                        xytext=(6, 2), textcoords="offset points",
                        fontsize=8)

    ax.set_title(
        f"Generation dispatch vs grid import by site -- {year}",
        fontsize=12)
    ax.set_xlabel("Generation dispatch (GWh/year)")
    ax.set_ylabel("Grid import (GWh/year)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = outdir / f"{year}_validation_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_validation_accounting(year: int,
                                outdir: Path) -> None:
    """
    Horizontal bar chart showing the five-check accounting of the
    generation dispatch vs grid import discrepancy.
    """
    # Values from the validation analysis
    total_gap      = 1992.0
    unmatched      =   60.3
    extra_pocs     =   45.2
    unexplained    = total_gap - unmatched - extra_pocs

    labels = [
        "Unmatched import sites\n(not in gen dispatch register)",
        "Extra sub-transmission POCs\nat generation sites",
        "Unexplained remainder\n(EA methodology)"
    ]
    values = [unmatched, extra_pocs, unexplained]
    colors = [COL_EXP, COL_IMP, "tomato"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} GWh",
                va="center", fontsize=9)

    ax.axvline(x=total_gap, color="gray", linewidth=1.0,
               linestyle="--", label=f"Total gap: {total_gap:.0f} GWh")
    ax.set_xlabel("Energy (GWh)")
    ax.set_title(
        f"Accounting for generation dispatch vs grid import gap -- {year}",
        fontsize=12)
    ax.legend()
    ax.set_xlim(0, total_gap * 1.25)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    path = outdir / f"{year}_validation_accounting.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_validation_site_diff(anndir: Path,
                               year: int,
                               outdir: Path,
                               top_n: int = 30) -> None:
    """
    Bar chart of annual difference (gen dispatch minus grid import) per site,
    sorted by magnitude, showing top_n sites.

    Positive values mean gen dispatch > import.
    Negative values mean import > gen dispatch (unusual).
    """
    gen_df = load_annual_df(anndir, year, "gen")
    imp_df = load_annual_df(anndir, year, "import")

    gen_tot = gen_df.sum() / 1000.0
    imp_tot = imp_df.sum() / 1000.0

    common = gen_tot.index.intersection(imp_tot.index)
    diff   = (gen_tot.loc[common] - imp_tot.loc[common]).sort_values(
        key=abs, ascending=False).head(top_n)

    colors = [COL_GEN if v >= 0 else COL_EXP for v in diff.values]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(diff)), diff.values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(diff)))
    ax.set_xticklabels(diff.index, rotation=90, fontsize=8)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.set_title(
        f"Site-level difference: gen dispatch minus grid import -- {year} "
        f"(top {top_n} by magnitude)",
        fontsize=12)
    ax.set_xlabel("Site")
    ax.set_ylabel("Difference (GWh/year)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = outdir / f"{year}_validation_site_diff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path.name}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot annual overview of NZ grid energy datasets."
    )
    parser.add_argument("--year",   type=int, required=True,
                        help="Year to plot, e.g. 2024")
    parser.add_argument("--anndir", default="./annual",
                        help="Directory containing annual CSV files "
                             "(default: ./annual)")
    parser.add_argument("--outdir", default="./plots",
                        help="Output directory for plots (default: ./plots)")
    parser.add_argument("--week",   default=None,
                        help="Week start date for weekly detail plots, "
                             "e.g. 2024-07-07 (default: first week of July)")

    # Plot group toggles -- all default to True
    parser.add_argument("--no-all-three",     action="store_true",
                        help="Skip three-dataset overview plots "
                             "(annual, monthly, weekly)")
    parser.add_argument("--no-import-export", action="store_true",
                        help="Skip import/export only plots "
                             "(model input plots)")
    parser.add_argument("--no-validation",    action="store_true",
                        help="Skip validation plots "
                             "(scatter, accounting, site diff)")

    args   = parser.parse_args()
    anndir = Path(args.anndir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_all_three     = not args.no_all_three
    plot_imp_exp       = not args.no_import_export
    plot_validation    = not args.no_validation

    # Determine which datasets to load
    need_gen = plot_all_three or plot_validation
    need_imp = plot_all_three or plot_imp_exp or plot_validation
    need_exp = plot_all_three or plot_imp_exp

    gen = load_annual(anndir, args.year, "gen")    if need_gen else None
    imp = load_annual(anndir, args.year, "import") if need_imp else None
    exp = load_annual(anndir, args.year, "export") if need_exp else None

    # --- Group A: all three datasets ----------------------------------------
    if plot_all_three:
        log.info("--- Plotting three-dataset overview ---")
        plot_annual_timeseries(gen, imp, exp, args.year, outdir)
        plot_monthly_bar(gen, imp, exp, args.year, outdir)
        plot_weekly_detail(gen, imp, exp, args.year, outdir,
                           week_start=args.week)

    # --- Group B: import/export only ----------------------------------------
    if plot_imp_exp:
        log.info("--- Plotting import/export model inputs ---")
        plot_annual_timeseries_imp_exp(imp, exp, args.year, outdir)
        plot_monthly_bar_imp_exp(imp, exp, args.year, outdir)
        plot_weekly_detail_imp_exp(imp, exp, args.year, outdir,
                                   week_start=args.week)

    # --- Group C: validation ------------------------------------------------
    if plot_validation:
        log.info("--- Plotting validation ---")
        plot_validation_scatter(anndir, args.year, outdir)
        plot_validation_accounting(args.year, outdir)
        plot_validation_site_diff(anndir, args.year, outdir)

    log.info("All plots complete.")


if __name__ == "__main__":
    main()
