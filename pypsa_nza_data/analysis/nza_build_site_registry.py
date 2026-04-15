# -*- coding: utf-8 -*-
"""
build_site_registry.py

Builds the definitive site registry for the PyPSA network by taking the
union of all sites appearing in the annual export and import files.
Merges with nodes.csv to add coordinates and metadata.

Usage:
    python build_site_registry.py --year 2024 --outdir ./annual
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build definitive site registry from annual files."
    )
    parser.add_argument("--year",    type=int, required=True,
                        help="Year to process, e.g. 2024")
    parser.add_argument("--anndir", default="./annual",
                        help="Directory containing annual CSV files")
    parser.add_argument("--nodes",  default="nodes.csv",
                        help="Path to nodes.csv")
    parser.add_argument("--outdir", default="./annual",
                        help="Output directory")
    args = parser.parse_args()

    anndir = Path(args.anndir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load annual files --------------------------------------------------
    exp_path = anndir / f"{args.year}_export_sites_all.csv"
    imp_path = anndir / f"{args.year}_import_sites_all.csv"

    log.info(f"Loading {exp_path.name}...")
    df_exp = pd.read_csv(exp_path, index_col=0, parse_dates=True)

    log.info(f"Loading {imp_path.name}...")
    df_imp = pd.read_csv(imp_path, index_col=0, parse_dates=True)

    # --- Build site sets ----------------------------------------------------
    exp_sites = set(df_exp.columns)
    imp_sites = set(df_imp.columns)
    all_sites = sorted(exp_sites | imp_sites)

    exp_only  = sorted(exp_sites - imp_sites)
    imp_only  = sorted(imp_sites - exp_sites)
    both      = sorted(exp_sites & imp_sites)

    log.info(f"Export sites (load):       {len(exp_sites)}")
    log.info(f"Import sites (generation): {len(imp_sites)}")
    log.info(f"Union (total buses):       {len(all_sites)}")
    log.info(f"  Load only:               {len(exp_only)}")
    log.info(f"  Generation only:         {len(imp_only)}")
    log.info(f"  Both:                    {len(both)}")

    # --- Build registry dataframe -------------------------------------------
    registry = pd.DataFrame(index=all_sites)
    registry.index.name = "site"
    registry["has_load"]       = registry.index.isin(exp_sites)
    registry["has_generation"] = registry.index.isin(imp_sites)

    # Add annual energy totals
    registry["load_GWh"]       = (df_exp.sum() / 1000.0).reindex(registry.index)
    registry["generation_GWh"] = (df_imp.sum() / 1000.0).reindex(registry.index)
    registry = registry.fillna(0.0)

    # --- Merge with nodes.csv -----------------------------------------------
    nodes_path = Path(args.nodes)
    if nodes_path.exists():
        log.info(f"Merging with {nodes_path.name}...")
        df_nodes = pd.read_csv(nodes_path, sep=None, engine="python")
        df_nodes = df_nodes.set_index("site")

        registry = registry.join(df_nodes, how="left")

        # Report sites missing from nodes.csv
        missing = registry[registry["lat"].isna()].index.tolist()
        if missing:
            log.warning(f"  Sites not found in nodes.csv: {missing}")
        else:
            log.info(f"  All sites found in nodes.csv")
    else:
        log.warning(f"nodes.csv not found at {nodes_path}")

    # --- Write output -------------------------------------------------------
    out_path = outdir / f"{args.year}_site_registry.csv"
    registry.to_csv(out_path)
    log.info(f"Site registry written to: {out_path}")

    # --- Summary ------------------------------------------------------------
    log.info("=== Site Registry Summary ===")
    log.info(f"  Total buses:       {len(registry)}")
    log.info(f"  Load only:         {len(exp_only)}")
    log.info(f"  Generation only:   {len(imp_only)}")
    log.info(f"  Both:              {len(both)}")
    log.info(f"  Total load:        {registry['load_GWh'].sum():.1f} GWh")
    log.info(f"  Total generation:  {registry['generation_GWh'].sum():.1f} GWh")
    log.info("Done.")


if __name__ == "__main__":
    main()