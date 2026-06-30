#!/usr/bin/env python3
"""
Standalone script to bootstrap grading scale mappings.

Usage:
    uv run scripts/bootstrap_grading_scales.py [--data-dir PATH] [--force-reload]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from chaksudb.db import close_pool
from chaksudb.ingest.framework.scale_bootstrap import bootstrap_grading_scales

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    """Main entry point for bootstrap script."""
    parser = argparse.ArgumentParser(
        description="Bootstrap grading scale mappings from SUSTech-SYSU dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to SUSTech-SYSU dataset directory (default: ./data/30_SUSTech-SYSU)",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force re-analysis even if mappings exist",
    )

    args = parser.parse_args()

    try:
        stats = await bootstrap_grading_scales(
            sustech_data_dir=args.data_dir,
            force_reload=args.force_reload,
        )

        print("\n✅ Bootstrap completed successfully!")
        print(f"   Total mappings: {stats['total_mappings']}")
        print(f"   Exact: {stats['exact_mappings']}")
        print(f"   Approximate: {stats['approximate_mappings']}")
        print(f"   Manual review: {stats['manual_review_required']}")

        return 0

    except Exception as e:
        print(f"\n❌ Bootstrap failed: {e}", file=sys.stderr)
        logging.exception("Bootstrap failed")
        return 1

    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
