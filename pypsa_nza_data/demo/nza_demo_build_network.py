from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pypsa
import yaml


# ---------------------------
# Demo noise suppression
# ---------------------------
def suppress_demo_warnings() -> None:
    import pandas as pd

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # PyPSA emits many UserWarnings for partial/demo networks
    warnings.filterwarnings("ignore", category=UserWarning, module="pypsa")

    # Pandas PerformanceWarning spam can occur during PyPSA export (fragmentation)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def configure_pypsa_logging() -> None:
    logging.getLogger("pypsa").setLevel(logging.ERROR)


# ---------------------------
# Config helpers (self-contained)
# ---------------------------
def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_and_merge_yaml(config_paths: List[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for p in config_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a YAML mapping/dict: {path}")
        merged = deep_update(merged, data)
    return merged


def require_cfg(cfg: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {dotted_key}")
        cur = cur[part]
    return cur


def resolve_under_root(root: Path, rel: str) -> Path:
    return (root / rel).resolve()


# ---------------------------
# Months parsing
# ---------------------------
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_months_arg(months: str) -> List[int]:
    m = months.strip().lower()
    if m == "all":
        return list(range(1, 13))

    if ".." in m:
        left, right = m.split("..", 1)
        left_i = int(left.strip())
        right_i = int(right.strip())
        if not (1 <= left_i <= 12 and 1 <= right_i <= 12 and left_i <= right_i):
            raise ValueError(f"Invalid month range: {months}")
        return list(range(left_i, right_i + 1))

    parts = [x.strip().lower() for x in m.split(",") if x.strip()]
    out: List[int] = []
    for p in parts:
        if p in _MONTHS:
            out.append(_MONTHS[p])
        else:
            out.append(int(p))
    out = sorted(set(out))
    if not out or any((mm < 1 or mm > 12) for mm in out):
        raise ValueError(f"Invalid --months value: {months}")
    return out


def _month_label(year: int, month: int) -> str:
    return f"{year}{month:02d}"


# ---------------------------
# Path model with discovery
# ---------------------------
@dataclass
class Paths:
    # dynamic (from config)
    demand_dir: Path
    gen_dir: Optional[Path]

    # discovered static / manual
    static_candidates: List[Path]
    manual_dir: Path

    # demo outputs
    out_demo_dir: Path
    out_network_dir: Path
    out_plot_dir: Path
    out_summary_dir: Path


def _resolve_paths(root: Path, cfg: Dict[str, Any]) -> Paths:
    # Demand dir from config (required)
    demand_dir = resolve_under_root(root, require_cfg(cfg, "paths.dirpath_d_MW"))

    # Optional gen dir from config (not required)
    gen_dir = None
    if isinstance(cfg.get("paths", {}), dict) and "dirpath_g_MW" in cfg["paths"]:
        gen_dir = resolve_under_root(root, cfg["paths"]["dirpath_g_MW"])

    # Processed static dir from config (often exists)
    processed_static = resolve_under_root(root, require_cfg(cfg, "paths.dirpath_static"))

    # Additional common static locations
    raw_static = (root / "data" / "raw" / "static").resolve()
    raw_static_csv = (raw_static / "csv").resolve()
    processed_static_csv = (processed_static / "csv").resolve()

    static_candidates = [processed_static, raw_static, raw_static_csv, processed_static_csv]

    manual_dir = (root / "data" / "manual").resolve()

    out_demo_dir = root / "demo"
    out_network_dir = out_demo_dir / "networks"
    out_plot_dir = out_demo_dir / "plots"
    out_summary_dir = out_demo_dir / "summaries"

    return Paths(
        demand_dir=demand_dir,
        gen_dir=gen_dir,
        static_candidates=static_candidates,
        manual_dir=manual_dir,
        out_demo_dir=out_demo_dir,
        out_network_dir=out_network_dir,
        out_plot_dir=out_plot_dir,
        out_summary_dir=out_summary_dir,
    )


def _find_required_file(filename: str, search_dirs: List[Path]) -> Path:
    for d in search_dirs:
        p = d / filename
        if p.exists():
            return p
    tried = "\n  - " + "\n  - ".join(str((d / filename)) for d in search_dirs)
    raise FileNotFoundError(f"Required file '{filename}' not found. Tried:{tried}")


# ---------------------------
# Data loading
# ---------------------------
def _read_csv_datetime_index(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _load_static_tables(cfg: Dict[str, Any], paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load topology tables.

    Buses + generators: searched in static_candidates
    Manual line/link mapping: searched in [manual_dir] + static_candidates (fallback)
    """
    bus_file = require_cfg(cfg, "filename.bus_data")
    line_file = require_cfg(cfg, "filename.line_data")
    link_file = require_cfg(cfg, "filename.link_data")
    gen_file = require_cfg(cfg, "filename.gen_data")

    static_search = paths.static_candidates
    manual_search = [paths.manual_dir] + paths.static_candidates

    bus_path = _find_required_file(bus_file, static_search)

    line_path = _find_required_file(line_file, manual_search)
    link_path = _find_required_file(link_file, manual_search)

    # gen_data.csv is a manual artifact in your workflow, so search manual first
    gen_search = [paths.manual_dir] + static_search
    gen_path = _find_required_file(gen_file, gen_search)

    df_bus = pd.read_csv(bus_path)
    df_line = pd.read_csv(line_path)
    df_link = pd.read_csv(link_path)
    df_gen = pd.read_csv(gen_path)

    return df_bus, df_line, df_link, df_gen



# ---------------------------
# Network build
# ---------------------------
def _create_base_topology(cfg: Dict[str, Any], paths: Paths) -> pypsa.Network:
    n = pypsa.Network()
    df_bus, df_line, df_link, df_gen = _load_static_tables(cfg, paths)

    # Buses: prefer "site" then "name"
    if "site" in df_bus.columns:
        bus_name_col = "site"
    elif "name" in df_bus.columns:
        bus_name_col = "name"
    else:
        raise KeyError("Bus table must contain a 'site' or 'name' column.")
#---------------------------------------------------------------------------------------------
    def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """
        Return the actual column name in df matching one of candidates,
        using case-insensitive comparison.
        """
        cols = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None


    # Accept common coordinate variants (case-insensitive)
    x_col = _pick_col(
        df_bus,
        [
            "x", "x_coord", "xcoord",
            "lon", "long", "longitude",
            "easting", "east",
        ],
    )

    y_col = _pick_col(
        df_bus,
        [
            "y", "y_coord", "ycoord",
            "lat", "latitude",
            "northing", "north",
        ],
    )

    if x_col is None or y_col is None:
        raise KeyError(
            "Bus table is missing coordinate columns. "
            "Expected X/Y or lat/long (case-insensitive). "
            f"Found columns: {list(df_bus.columns)}"
        )

#---------------------------------------------------------------------------------------------
    for _, row in df_bus.iterrows():
        name = str(row[bus_name_col])
        n.add(
            "Bus",
            name=name,
            x=float(row[x_col]),
            y=float(row[y_col]),
            carrier=row.get("carrier", "AC"),
            v_nom=row.get("v_nom", np.nan),
        )

    # Lines: require bus0/bus1 columns (common in your manual mapping)
    if not {"bus0", "bus1"}.issubset(df_line.columns):
        raise KeyError("Line table must contain 'bus0' and 'bus1' columns.")
    for i, row in df_line.iterrows():
        name = str(row.get("name", row.get("line", f"line_{i}")))
        n.add(
            "Line",
            name=name,
            bus0=str(row["bus0"]),
            bus1=str(row["bus1"]),
            s_nom=row.get("s_nom", np.nan),
            r=row.get("r", np.nan),
            x=row.get("x", np.nan),
            length=row.get("length", np.nan),
        )

    # Links (HVDC etc.)
    if not {"bus0", "bus1"}.issubset(df_link.columns):
        raise KeyError("Link table must contain 'bus0' and 'bus1' columns.")
    for i, row in df_link.iterrows():
        name = str(row.get("name", row.get("link", f"link_{i}")))
        n.add(
            "Link",
            name=name,
            bus0=str(row["bus0"]),
            bus1=str(row["bus1"]),
            p_nom=row.get("p_nom", np.nan),
            efficiency=row.get("efficiency", 1.0),
            carrier=row.get("carrier", "HVDC"),
        )

    # Generators metadata (dormant)
    for i, row in df_gen.iterrows():
        site = str(row["site"]) if "site" in df_gen.columns else str(row.get("name", f"gen_{i}"))
        n.add(
            "Generator",
            name=site,
            bus=site,
            p_nom=row.get("p_nom", np.nan),
            carrier=row.get("carrier", "unknown"),
            build_year=row.get("build_year", np.nan),
            efficiency=row.get("efficiency", 1.0),
            p_max_pu=0.0,
            p_min_pu=0.0,
        )

    return n


def _build_month_network(base: pypsa.Network, demand_file: Path) -> pypsa.Network:
    df_d = _read_csv_datetime_index(demand_file)

    n = base.copy()
    n.set_snapshots(df_d.index)

    bus_names = set(n.buses.index.astype(str))
    added = 0
    skipped = 0

    for col in df_d.columns:
        site = str(col)
        if site not in bus_names:
            skipped += 1
            continue
        p = df_d[site].to_numpy()
        n.add("Load", name=site, bus=site, p_set=p)
        added += 1

    n.meta = dict(n.meta or {})
    n.meta["demand_file"] = str(demand_file)
    n.meta["loads_added"] = added
    n.meta["loads_skipped"] = skipped

    return n


# ---------------------------
# Outputs
# ---------------------------
def _simple_network_plot(n: pypsa.Network, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    buses = n.buses.copy()
    if "x" not in buses.columns or "y" not in buses.columns:
        # If coords are missing, skip plot rather than fail demo
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")

    ax.scatter(buses["x"].values, buses["y"].values, s=6)

    if len(n.lines) > 0:
        for _, row in n.lines.iterrows():
            b0, b1 = row["bus0"], row["bus1"]
            if b0 in buses.index and b1 in buses.index:
                x0, y0 = buses.loc[b0, "x"], buses.loc[b0, "y"]
                x1, y1 = buses.loc[b1, "x"], buses.loc[b1, "y"]
                ax.plot([x0, x1], [y0, y1], linewidth=0.5)

    if len(n.links) > 0:
        for _, row in n.links.iterrows():
            b0, b1 = row["bus0"], row["bus1"]
            if b0 in buses.index and b1 in buses.index:
                x0, y0 = buses.loc[b0, "x"], buses.loc[b0, "y"]
                x1, y1 = buses.loc[b1, "x"], buses.loc[b1, "y"]
                ax.plot([x0, x1], [y0, y1], linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    ax.set_xticks([])
    ax.set_yticks([])


    ax.autoscale(enable=True, tight=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary(n: pypsa.Network, out_txt: Path) -> Tuple[float, float]:
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # 30-min MW -> MWh
    total_mwh = float(np.nansum(n.loads_t.p_set.values)) * 0.5 if len(n.loads) else 0.0
    peak_mw = float(np.nanmax(n.loads_t.p_set.values)) if len(n.loads) and n.loads_t.p_set.size else 0.0

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("PyPSA NZ demo network summary\n")
        f.write("--------------------------------\n")
        if len(n.snapshots):
            f.write(f"Snapshots: {n.snapshots.min()} .. {n.snapshots.max()} (n={len(n.snapshots)})\n")
        f.write(f"Buses: {len(n.buses)}\n")
        f.write(f"Lines: {len(n.lines)}\n")
        f.write(f"Links: {len(n.links)}\n")
        f.write(f"Generators: {len(n.generators)}\n")
        f.write(f"Loads: {len(n.loads)}\n")
        f.write(f"Total demand (GWh): {total_mwh / 1000.0:.3f}\n")
        f.write(f"Peak demand (MW): {peak_mw:.3f}\n")
        if hasattr(n, "meta") and isinstance(n.meta, dict):
            f.write(f"Loads added: {n.meta.get('loads_added', '')}\n")
            f.write(f"Loads skipped (no matching bus): {n.meta.get('loads_skipped', '')}\n")
            f.write(f"Demand file: {n.meta.get('demand_file', '')}\n")

    return total_mwh, peak_mw


# ---------------------------
# Main
# ---------------------------
def main(argv: Optional[List[str]] = None) -> int:
    suppress_demo_warnings()
    configure_pypsa_logging()

    ap = argparse.ArgumentParser(
        prog="python -m pypsa_nza_data.demo.nza_demo_build_network",
        description="Demo: build PyPSA networks (NetCDF) for NZ from processed outputs.",
    )
    ap.add_argument("--root", required=True, help="Workspace root directory (external).")
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config path (can be provided multiple times; later overrides earlier).",
    )
    ap.add_argument("--year", type=int, default=2024, help="Year to build (default: 2024).")
    ap.add_argument(
        "--months",
        default="all",
        help="Months to build: all | jan | jan,feb | 1..12 | 1,2,12 (default: all).",
    )
    ap.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export PyPSA CSV folders for each month (debugging only).",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not args.config:
        raise SystemExit("At least one --config YAML must be provided.")

    cfg = load_and_merge_yaml(args.config)
    paths = _resolve_paths(root, cfg)

    months = parse_months_arg(args.months)

    print(f"Building PyPSA demo networks for year={args.year}, months={months}")
    print(f"Demand dir: {paths.demand_dir}")
    print(f"Manual dir: {paths.manual_dir}")
    print("Static candidates:")
    for d in paths.static_candidates:
        print(f"  - {d} (exists={d.exists()})")
    print(f"Outputs: {paths.out_demo_dir}")

    # Build base topology once
    base = _create_base_topology(cfg, paths)

    annual_mwh = 0.0
    annual_peak_mw = 0.0

    for m in months:
        label = _month_label(args.year, m)
        demand_file = paths.demand_dir / f"{label}_d_MW.csv"
        if not demand_file.exists():
            raise FileNotFoundError(f"Missing demand file: {demand_file}")

        n = _build_month_network(base, demand_file)

        # Write artifacts
        paths.out_network_dir.mkdir(parents=True, exist_ok=True)
        nc_path = paths.out_network_dir / f"{label}.nc"
        n.export_to_netcdf(nc_path)

        summary_path = paths.out_summary_dir / f"{label}_summary.txt"
        month_mwh, month_peak_mw = _write_summary(n, summary_path)

        # Topology is invariant across months: write one plot only (first month)
        if m == months[0]:
            plot_path = paths.out_plot_dir / f"{args.year}_topology.png"
            _simple_network_plot(n, plot_path, title=f"NZ PyPSA demo network topology ({args.year})")


        if args.export_csv:
            csv_dir = paths.out_network_dir / f"{label}_csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            n.export_to_csv_folder(str(csv_dir))

        annual_mwh += month_mwh
        annual_peak_mw = max(annual_peak_mw, month_peak_mw)

        print(f"  * {label}: saved {nc_path.name}")

    rollup = paths.out_summary_dir / f"{args.year}_annual_rollup.txt"
    rollup.parent.mkdir(parents=True, exist_ok=True)
    with rollup.open("w", encoding="utf-8") as f:
        f.write(f"Annual rollup for {args.year}\n")
        f.write("-------------------------\n")
        f.write(f"Months built: {months}\n")
        f.write(f"Total demand (GWh): {annual_mwh / 1000.0:.3f}\n")
        f.write(f"Peak demand (MW): {annual_peak_mw:.3f}\n")

    print(f"? Annual rollup written: {rollup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
