"""
View Routing Metrics Report.

Queries the PostgreSQL database and computes evaluation metrics
from production routing decisions.

Usage:
    python -m ignis_router.evaluation.report              # Last 24 hours
    python -m ignis_router.evaluation.report --days 7     # Last 7 days
    python -m ignis_router.evaluation.report --days 30    # Last 30 days
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Ignis Router - Metrics Report")
    parser.add_argument("--days", type=int, default=1, help="Number of days to report on (default: 1)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of summary")
    args = parser.parse_args()

    # Load .env from project root
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")

    from .metrics import MetricsEngine

    engine = MetricsEngine()
    report = engine.compute(days=args.days)

    if report.total_queries == 0:
        print(f"No routing decisions found in the last {args.days} day(s).")
        print("Run some queries through the router first (e.g., python examples/ai_chat_app.py)")
        return

    if args.json:
        import json
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
