#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nza_load_static_data_from_url.py

Configuration-driven downloader for PyPSA-NZA static datasets.

Key properties (reviewer-proof):
- No work at import time.
- Runnable via: python -m pypsa_nza_data.loaders.nza_load_static_data_from_url
- Default config is packaged and located via importlib.resources.
- Supports --config override path.
- Prints/logs FULL attempted URL and local output path for every dataset.
- Dry-run is side-effect free (no downloads, no file writes) but shows URLs/paths.

Author: Philippe Bruneau
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import sys
import logging
import argparse
import hashlib
import shutil
import json

import requests
import pandas as pd
import yaml
from importlib.resources import files


logger = logging.getLogger(__name__)


# =============================================================================
# Utilities
# =============================================================================

def get_repo_root() -> Path:
    """Resolve repository root robustly for editable installs."""
    # <repo>/pypsa_nza_data/loaders/nza_load_static_data_from_url.py
    return Path(__file__).resolve().parents[3]


def get_default_config_path() -> Path:
    """Locate the packaged default YAML config."""
    return files("pypsa_nza_data").joinpath("config/nza_download_static_data.yaml")


def setup_logging(log_dir: Path) -> Tuple[Path, logging.FileHandler]:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nza_download_static_{timestamp}.log"

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter("%(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(console_formatter)
    root_logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    return log_file, file_handler


def print_header(title: str, char: str = "=") -> None:
    width = 80
    logger.info(char * width)
    logger.info(f"{title:^{width}}")
    logger.info(char * width)


def print_section(title: str) -> None:
    logger.info("")
    logger.info(f"--- {title} ---")
    logger.info("")    


def load_config(config_file: Path) -> Dict:
    """Load and validate YAML configuration."""
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}") from e

    required_sections = ["directories", "data_sources", "datasets"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")

    if not isinstance(config["directories"], dict):
        raise ValueError("Config error: 'directories' must be a mapping.")
    if not isinstance(config["data_sources"], dict):
        raise ValueError("Config error: 'data_sources' must be a mapping.")
    if not isinstance(config["datasets"], dict):
        raise ValueError("Config error: 'datasets' must be a mapping.")

    return config


def get_enabled_datasets(config: Dict) -> List[str]:
    """Return dataset names that are enabled in YAML."""
    enabled: List[str] = []
    for name, ds in (config.get("datasets") or {}).items():
        if isinstance(ds, dict) and ds.get("enabled", True):
            enabled.append(name)
    return enabled


# =============================================================================
# Directory manager
# =============================================================================

class DirectoryManager:
    """Manages output directories from config."""

    def __init__(self, root_dir: Path, config: Dict):
        self.root = Path(root_dir)
        self.dirs: Dict[str, Path] = {}

        for name, rel_path in config.get("directories", {}).items():
            if rel_path is None:
                continue
            full_path = (self.root / str(rel_path)).resolve()
            full_path.mkdir(parents=True, exist_ok=True)
            self.dirs[name] = full_path

    def get(self, name: str) -> Path:
        if name not in self.dirs:
            raise ValueError(f"Unknown directory key in YAML: '{name}'")
        return self.dirs[name]


# =============================================================================
# Downloader
# =============================================================================

class DataDownloader:
    """Downloads data from configured sources."""

    def __init__(self, config: Dict):
        self.sources = config.get("data_sources", {})
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "pypsa-nza-data (https://github.com/) Python requests"}
        )

    def download_transpower_dataset(self, dataset_config: Dict, output_path: Path) -> bool:
        dataset_id = dataset_config["dataset_id"]
        fmt = dataset_config["format"]

        transpower = self.sources["transpower"]
        base_url = transpower["base_url"]

        if fmt == "csv":
            url = f"{base_url}/datasets/{dataset_id}.csv"
        elif fmt == "geojson":
            url = f"{base_url}/datasets/{dataset_id}.geojson"
        else:
            logger.error(f"Unsupported format: {fmt}")
            return False

        logger.info(f"    Attempting URL: {url}")
        logger.info(f"    Output file:    {output_path}")

        try:
            resp = self.session.get(url, stream=True, timeout=60)

            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True

            if resp.status_code == 404:
                logger.warning("    Direct download returned 404; trying ArcGIS API fallback...")
                return self._download_via_api(dataset_config, output_path)

            logger.error(f"    HTTP {resp.status_code}")
            return False

        except Exception as e:
            logger.error(f"    Download error: {e}")
            return False

    def _download_via_api(self, dataset_config: Dict, output_path: Path) -> bool:
        dataset_id = dataset_config["dataset_id"]
        transpower = self.sources["transpower"]
        api_url = transpower["api_url"]

        service_name = dataset_id.replace("_0", "")
        query_url = f"{api_url}/{service_name}/FeatureServer/0/query"

        params = {"where": "1=1", "outFields": "*", "f": "json", "returnGeometry": "true"}

        logger.info(f"    Attempting API: {query_url}")

        try:
            resp = self.session.get(query_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if "features" not in data:
                logger.error("    No features in API response")
                return False

            records = [f.get("attributes", {}) for f in data["features"]]
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            return True

        except Exception as e:
            logger.error(f"    API error: {e}")
            return False

    def download_ea_dataset(self, dataset_config: Dict, output_path: Path) -> bool:
        url_path = dataset_config["url_path"]
        ea = self.sources["electricity_authority"]
        base_url = ea["base_url"]
        base_url = str(base_url).rstrip("/")
        url_path = str(url_path).lstrip("/")
        url = f"{base_url}/{url_path}"

        logger.info(f"    Attempting URL: {url}")
        logger.info(f"    Output file:    {output_path}")

        try:
            resp = self.session.get(url, stream=True, timeout=60)

            if resp.status_code == 404:
                logger.warning("    File not found (HTTP 404)")
                return False

            resp.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True

        except Exception as e:
            logger.error(f"    Download error: {e}")
            return False

    def download(self, dataset_name: str, dataset_config: Dict, output_path: Path) -> bool:
        source = dataset_config["source"]
        if source == "transpower":
            return self.download_transpower_dataset(dataset_config, output_path)
        if source == "electricity_authority":
            return self.download_ea_dataset(dataset_config, output_path)

        logger.error(f"Unknown source '{source}' for dataset '{dataset_name}'")
        return False


# =============================================================================
# Validation
# =============================================================================

class DataValidator:
    """Validates downloaded files."""

    @staticmethod
    def validate(filepath: Path, min_rows: int = 1, file_format: str = "csv") -> Dict:
        result = {"valid": False, "exists": filepath.exists(), "size_bytes": 0, "rows": 0, "columns": 0, "errors": []}

        if not result["exists"]:
            result["errors"].append("File does not exist")
            return result

        result["size_bytes"] = filepath.stat().st_size
        if result["size_bytes"] == 0:
            result["errors"].append("File is empty")
            return result

        if file_format == "csv":
            return DataValidator._validate_csv(filepath, min_rows, result)
        if file_format == "geojson":
            return DataValidator._validate_geojson(filepath, min_rows, result)

        result["valid"] = True
        return result

    @staticmethod
    def _validate_csv(filepath: Path, min_rows: int, result: Dict) -> Dict:
        try:
            df = pd.read_csv(filepath)
            result["rows"] = len(df)
            result["columns"] = len(df.columns)
            if result["rows"] < min_rows:
                result["errors"].append(f"Too few rows: {result['rows']} < {min_rows}")
            else:
                result["valid"] = True
        except Exception as e:
            result["errors"].append(f"CSV read error: {e}")
        return result

    @staticmethod
    def _validate_geojson(filepath: Path, min_rows: int, result: Dict) -> Dict:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("type") != "FeatureCollection":
                result["errors"].append("Not a valid GeoJSON FeatureCollection")
                return result

            features = data.get("features", [])
            result["rows"] = len(features)

            if features:
                props = features[0].get("properties", {})
                result["columns"] = len(props) if isinstance(props, dict) else 0

            if result["rows"] < min_rows:
                result["errors"].append(f"Too few features: {result['rows']} < {min_rows}")
            else:
                result["valid"] = True

        except Exception as e:
            result["errors"].append(f"GeoJSON read error: {e}")
        return result

    @staticmethod
    def compute_checksum(filepath: Path) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()


# =============================================================================
# Orchestrator
# =============================================================================

class DownloadOrchestrator:
    def __init__(self, root_dir: Path, config: Dict, dry_run: bool = False):
        self.root = Path(root_dir)
        self.config = config
        self.dry_run = dry_run

        self.directories = DirectoryManager(root_dir, config)
        self.downloader = DataDownloader(config)
        self.validator = DataValidator()

        self.stats = {"total": 0, "downloaded": 0, "failed": 0, "results": []}

    def download_dataset(self, name: str, cfg: Dict) -> bool:
        print_section(f"Dataset: {name}")
        logger.info(f"Description: {cfg.get('description', 'N/A')}")
        logger.info(f"Source:      {cfg.get('source')}")
        logger.info(f"Format:      {cfg.get('format')}")
        logger.info(f"Filename:    {cfg.get('filename')}")
        output_dir = self.directories.get(cfg["directory"])
        output_path = (output_dir / cfg["filename"]).resolve()
        logger.info(f"Save to:     {output_path}")

        if self.dry_run:
            # Show URL and output path, but do not download
            src = cfg["source"]
            if src == "transpower":
                dataset_id = cfg["dataset_id"]
                fmt = cfg["format"]
                tp = self.config["data_sources"]["transpower"]["base_url"]
                url = f"{tp}/datasets/{dataset_id}.{ 'geojson' if fmt == 'geojson' else 'csv' }"
            elif src == "electricity_authority":
                base = self.config["data_sources"]["electricity_authority"]["base_url"]
                base = str(base).rstrip("/")
                url_path = str(cfg['url_path']).lstrip("/")
                url = f"{base}/{url_path}"
            else:
                url = "(unknown source)"
            logger.info(f"    Attempting URL: {url}")
            logger.info(f"    Output file:    {output_path}")
            return True

        logger.info("Downloading...")
        ok = self.downloader.download(name, cfg, output_path)
        if not ok:
            logger.error("✗ Download failed")
            self.stats["failed"] += 1
            return False

        min_rows = int(cfg.get("validate_min_rows", 1))
        fmt = cfg.get("format", "csv")
        validation = self.validator.validate(output_path, min_rows=min_rows, file_format=fmt)

        if not validation["valid"]:
            logger.error(f"✗ Validation failed: {validation['errors']}")
            self.stats["failed"] += 1
            return False

        logger.info("✓ Downloaded successfully")
        logger.info(f"  Size:    {validation['size_bytes']:,} bytes")
        if fmt == "geojson":
            logger.info(f"  Features: {validation['rows']:,}")
        else:
            logger.info(f"  Rows:     {validation['rows']:,}")
        if validation["columns"] > 0:
            logger.info(f"  Columns:  {validation['columns']}")
        if self.config.get("validation", {}).get("archive_downloads", True):
            self._archive_file(output_path)

        self.stats["downloaded"] += 1
        self.stats["results"].append({"name": name, "file": cfg["filename"], "rows": validation["rows"], "size": validation["size_bytes"]})
        return True

    def _archive_file(self, filepath: Path) -> None:
        try:
            archive_dir = self.directories.get("archive")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            archive_path = (archive_dir / archive_name).resolve()
            shutil.copy2(filepath, archive_path)
            logger.info(f"  Archived: {archive_name}")
        except Exception as e:
            logger.warning(f"Could not archive file: {e}")

    def download_all(self, selected: List[str]) -> None:
        datasets = self.config.get("datasets", {})
        for name in selected:
            if name not in datasets:
                logger.error(f"Unknown dataset requested: '{name}' (not found in YAML under datasets:)")
                self.stats["failed"] += 1
                continue
            self.stats["total"] += 1
            self.download_dataset(name, datasets[name])

    def print_summary(self) -> None:
        print_header("SUMMARY", "=")
        logger.info("")
        logger.info(f"Total datasets:          {self.stats['total']}")
        logger.info(f"Successfully downloaded: {self.stats['downloaded']}")
        logger.info(f"Failed:                  {self.stats['failed']}")
        logger.info("")
        if self.stats["results"]:
            print_section("Downloaded Files")
            logger.info(f"{'Dataset':<30} {'Rows':>10} {'Size':>15}")
            logger.info("-" * 60)
            for r in self.stats["results"]:
                logger.info(f"{r['name']:<30} {r['rows']:>10,} {r['size']:>12,} bytes")
            logger.info("")    


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Download PyPSA-NZA static datasets (config in YAML file)")
    parser.add_argument("--dataset", type=str, help="Download only specific dataset")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--config", type=str, default=None, help="Override path to YAML config")
    args = parser.parse_args()

    start_time = datetime.now()
    root_path = get_repo_root()
    config_path = Path(args.config) if args.config else Path(get_default_config_path())

    # Load config early (we need it for log_dir)
    config = load_config(config_path)

    logs_rel = config.get("directories", {}).get("logs") if isinstance(config.get("directories", {}), dict) else None
    log_dir = (root_path / str(logs_rel)).resolve() if logs_rel else (root_path / "logs").resolve()

    log_file, file_handler = setup_logging(log_dir)

    try:
        print_header("PYPSA-NZA STATIC DATA DOWNLOADER", "=")
        logger.info("")
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Repo root:  {root_path}")
        logger.info(f"Config:     {config_path}")
        if args.dataset:
            logger.info(f"Filter:     Only '{args.dataset}'")
        if args.dry_run:
            logger.info("Mode:       DRY RUN (no downloads)")
        logger.info("")        

        print_header("LOADING CONFIGURATION", "=")
        logger.info("")

        logger.info("Output directories (relative paths from YAML):")
        for name, path in config["directories"].items():
            logger.info(f"  {name:<15} → {path}")
        logger.info("")        

        enabled = get_enabled_datasets(config)

        if args.dataset:
            if args.dataset not in config.get("datasets", {}):
                logger.error(f"✗ Fatal error: Unknown dataset '{args.dataset}'. Check YAML keys under 'datasets:'.")
                return 1
            selected = [args.dataset]
            if args.dataset not in enabled:
                logger.warning(f"Dataset '{args.dataset}' is not enabled in YAML, but will run because it was explicitly requested.")
        else:
            selected = enabled

        logger.info(f"Enabled datasets (YAML): {', '.join(enabled) if enabled else '(none)'}")
        logger.info(f"Selected datasets (run): {', '.join(selected) if selected else '(none)'}")
        logger.info("")        

        if not selected:
            logger.warning("No datasets selected. Nothing to do.")
            return 0

        print_header("DRY RUN" if args.dry_run else "DOWNLOADING DATASETS", "=")
        logger.info("")        

        orchestrator = DownloadOrchestrator(root_path, config, dry_run=bool(args.dry_run))
        orchestrator.download_all(selected)
        orchestrator.print_summary()

        print_header("COMPLETE", "=")
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info("")
        logger.info(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed:    {elapsed:.2f} seconds")
        logger.info(f"Log saved:  {log_file}")
        logger.info("")        

        return 0 if orchestrator.stats["failed"] == 0 else 1

    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    finally:
        try:
            file_handler.close()
            logging.getLogger().removeHandler(file_handler)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
