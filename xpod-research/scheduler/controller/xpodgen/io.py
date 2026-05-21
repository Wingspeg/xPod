import csv
import logging
from typing import Dict, Iterable

logger = logging.getLogger(__name__)


def parse_csv(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def split_csv_list(s: str):
    return [x.strip() for x in (s or "").split(",") if x.strip()]
