#!/usr/bin/env python3
import argparse
import logging
import sys

from xpodgen import cli

logger = logging.getLogger(__name__)

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None)
    args, rest = ap.parse_known_args(argv)

    from scheduler.logging_config import setup_logging

    setup_logging(args.log_level, args.log_file)
    logger.info("main start args=%s", {"log_level": args.log_level, "log_file": args.log_file, "argv": rest})

    return int(cli.main(rest))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
