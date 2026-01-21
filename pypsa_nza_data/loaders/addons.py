def get_repo_root() -> Path:
    """
    Resolve repo root robustly for editable installs.
    This file is: pypsa_nza_data/loaders/<script>.py
    parents[2] => pypsa_nza_data/
    parents[3] => repo root
    """
    return Path(__file__).resolve().parents[3]


def get_default_config_path() -> Path:
    """
    Locate the packaged default YAML config.
    Ensure you place the YAML at: pypsa_nza_data/config/nza_download_dynamic_data.yaml
    """
    return files("pypsa_nza_data").joinpath("config/nza_download_dynamic_data.yaml")



def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description='Download PyPSA-NZA time-series datasets'
    )
    parser.add_argument('--dataset', type=str, help='Download only specific dataset')
    parser.add_argument('--years', type=int, nargs='+', help='Override years')
    parser.add_argument('--months', type=str, nargs='+', help='Override months')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (optional override)')

    args = parser.parse_args()
    start_time = datetime.now()

    # Resolve root + config path (reviewer-proof)
    root_path = get_repo_root()
    config_file = Path(args.config) if args.config else Path(get_default_config_path())

    # Load config early so we can derive log directory from it
    config = load_config(config_file)

    # Log directory: use YAML directories section if it has a logs entry; else default to <repo>/logs
    logs_rel = None
    if isinstance(config.get("directories", {}), dict):
        logs_rel = config["directories"].get("logs")

    log_dir = (root_path / logs_rel).resolve() if logs_rel else (root_path / "logs").resolve()
    log_file, file_handler = setup_logging(log_dir)

    # Header
    print_header("PYPSA-NZA DYNAMIC DATA DOWNLOADER")
    logger.info("")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config:     {config_file}")
    if args.dataset:
        logger.info(f"Filter:     {args.dataset}")
    if args.years:
        logger.info(f"Years:      {args.years}")
    if args.months:
        logger.info(f"Months:     {args.months}")
    logger.info("")
