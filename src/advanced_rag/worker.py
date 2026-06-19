from __future__ import annotations

import argparse
import time

from advanced_rag.config import get_settings
from advanced_rag.database import Database
from advanced_rag.ingestion import IngestionService


def run(once: bool = False, poll_seconds: float = 2.0) -> None:
    settings = get_settings()
    database = Database(settings.database_path)
    service = IngestionService(settings, database)
    while True:
        processed = service.process_next()
        if once:
            return
        if processed is None:
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process queued Advanced RAG documents")
    parser.add_argument("--once", action="store_true", help="Process at most one job")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    run(once=args.once, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
