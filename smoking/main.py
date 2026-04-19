from __future__ import annotations

import argparse
import time

from agent.manager import AgentManager
from discovery.service import run_interactive_camera_setup
from utils.config import build_default_raw_config, load_config, load_raw_config
from utils.logger import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stealth Lens Agent")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Caminho para o arquivo de configuração JSON.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Descobre câmeras automaticamente e permite escolher uma para salvar no config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_config = load_raw_config(args.config) if args.config else build_default_raw_config()
    logging_config = raw_config.get("logging", {"level": "INFO", "json": True})
    configure_logging(
        level=str(logging_config.get("level", "INFO")),
        json_output=bool(logging_config.get("json", True)),
    )

    logger = get_logger("stealth_lens.main")

    if args.discover:
        logger.info("starting_camera_discovery", extra={"config_path": args.config})
        discovery_result = run_interactive_camera_setup(args.config, logger)
        if not discovery_result.configured or not discovery_result.run_after_setup:
            return 0

    config = load_config(args.config)
    logger.info(
        "booting_stealth_lens_agent",
        extra={"agent_id": config.agent_id, "config_path": args.config},
    )

    manager = AgentManager(config)
    manager.start()

    try:
        while not manager.should_stop:
            time.sleep(0.25)
    except KeyboardInterrupt:
        logger.info("shutdown_requested", extra={"agent_id": config.agent_id})
    finally:
        manager.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
