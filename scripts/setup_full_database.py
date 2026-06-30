#!/usr/bin/env python3
"""
Master script to set up the full database by:
1. Bootstrapping grading scales
2. Running all 36 ingest scripts concurrently

Usage:
    uv run scripts/setup_full_database.py [--skip-bootstrap]
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from chaksudb.common.progress import OperationStatistics
from chaksudb.config.config import db_config
from chaksudb.db.connection import close_pool, init_pool
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid
from chaksudb.ingest.framework.provenance import reconcile_grade_conversions
from chaksudb.ingest.framework.scale_bootstrap import bootstrap_grading_scales
from chaksudb.ingest.framework.validation import (
    get_dataset_stats,
    validate_all_foreign_keys,
    validate_dataset,
)
from chaksudb.ingest.scripts.run_roi_iqa import run as run_roi_iqa

# Import all ingest functions
from chaksudb.ingest.scripts import *

import os

# Sibling bootstrap scripts live in this same `scripts/` directory.
sys.path.insert(0, str(Path(__file__).parent))
from bootstrap_dictionary_vocab import bootstrap_dictionary_vocab

# Suppress OpenCV TIFF warnings (harmless but noisy)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# File paths
full_log = log_dir / f"full_{timestamp}.log"
warning_log = log_dir / f"warnings_{timestamp}.log"
error_log = log_dir / f"errors_{timestamp}.log"

# Format used everywhere
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

# ---- Handlers ----

# 1. Everything log
full_handler = logging.FileHandler(full_log, encoding="utf-8")
full_handler.setLevel(logging.INFO)
full_handler.setFormatter(formatter)

# 2. Warnings only
warning_handler = logging.FileHandler(warning_log, encoding="utf-8")
warning_handler.setLevel(logging.WARNING)
warning_handler.setFormatter(formatter)

# 3. Errors only (includes exceptions with stacktrace)
error_handler = logging.FileHandler(error_log, encoding="utf-8")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# 4. Console output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# ---- Root logger ----
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        full_handler,
        warning_handler,
        error_handler,
        console_handler,
    ],
)

logger = logging.getLogger(__name__)

# Map of dataset names to their ingest functions
INGEST_FUNCTIONS = [
    ("EYEPACS", ingest_eyepacs),
    ("Messidor", ingest_messidor),
    ("IDRID", ingest_idrid),
    ("RFMiD", ingest_rfmid),
    ("1000x39", ingest_1000x39),
    ("DeepEyeNet", ingest_deepeyenet),
    ("LAG", ingest_lag),
    ("ODIR-5K", ingest_odir5k),
    ("PAPILA", ingest_papila),
    ("Paraguay", ingest_paraguay),
    ("STARE", ingest_stare),
    ("ARIA", ingest_aria),
    ("FIVES", ingest_fives),
    ("AGAR300", ingest_agar300),
    ("APTOS", ingest_aptos),
    ("Fund-OCT", ingest_fund_oct),
    ("DIARETDB1", ingest_diaretdb1),
    ("DRIONS-DB", ingest_drionsdb),
    ("Drishti-GS1", ingest_drishti_gs1),
    ("EOPHTA", ingest_eophta),
    ("G1020", ingest_g1020),
    ("HRF", ingest_hrf),
    ("ORIGA", ingest_origa),
    ("REFUGE", ingest_refuge),
    ("ROC", ingest_roc),
    ("BRSET", ingest_brset),
    ("OIA-DDR", ingest_oia_ddr),
    ("SUSTech-SYSU", ingest_sustech_sysu),
    ("JICHI", ingest_jichi),
    ("Chaksu", ingest_chaksu),
    ("DR1-2", ingest_dr1_2),
    ("Cataract", ingest_cataract),
    ("SCARDAT", ingest_scardat),
    ("ACRIMA", ingest_acrima),
    ("DeepDRID", ingest_deepdrid),
    ("MMAC", ingest_mmac),
    ("HEI-MED", ingest_hei_med),
    ("justRAIGS", ingest_justraigs),
    ("RFMID2", ingest_rfmid2),
    ("CHASEDB1", ingest_chasedb1),
    ("DRIVE", ingest_drive),
    ("DDR", ingest_ddr),
    ("RIM-ONE-DL", ingest_rim_one),
    ("RITE", ingest_rite),
    ("MuReD", ingest_mured),
    ("MESSIDOR2", ingest_messidor2),
    ("MBRSET", ingest_mbrset),
    ("AV-DRIVE", ingest_av_drive),
    ("Fundus-AVSeg", ingest_fundus_avseg),
    ("HRF-v1", ingest_hrf_v1),
    ("HRF-v2", ingest_hrf_v2),
    ("LES-AV", ingest_les_av),
    ("MAPLES-DR", ingest_maples_dr),
]


async def ingest_with_summary(
    dataset_name: str, ingest_func
) -> Tuple[str, Optional[OperationStatistics], Optional[Exception]]:
    """
    Wrapper function that calls an ingest function and logs a consistent summary.
    
    Args:
        dataset_name: Name of the dataset being ingested
        ingest_func: The async ingest function to call
        
    Returns:
        Tuple of (dataset_name, OperationStatistics, Exception if any)
    """
    logger.info("=" * 80)
    logger.info(f"Starting ingestion: {dataset_name}")
    logger.info("=" * 80)
    
    try:
        stats = await ingest_func()
        
        # Log consistent summary format (matching stare/aria scripts)
        logger.info("=" * 80)
        logger.info(f"Ingestion Summary: {dataset_name}")
        logger.info(f"  Total items: {stats.total_items}")
        logger.info(f"  Successful: {stats.successful_items}")
        logger.info(f"  Failed: {stats.failed_items}")
        logger.info(f"  Skipped: {stats.skipped_items}")
        
        # Show breakdown by type (using item_counts)
        if stats.item_counts:
            logger.info("")
            logger.info("  Breakdown by type:")
            for item_type, count in sorted(stats.item_counts.items()):
                logger.info(f"    {item_type}: {count}")
        
        if stats.errors:
            logger.warning(f"  Total errors: {len(stats.errors)}")
            # Group errors by type
            error_types = {}
            for error in stats.errors:
                error_type = error.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_types.items()):
                logger.warning(f"    {error_type}: {count}")
        else:
            logger.info("  No errors encountered")
        
        logger.info("=" * 80)
        
        if stats.failed_items > 0:
            logger.warning(f"{dataset_name} completed with errors")
        else:
            logger.info(f"{dataset_name} completed successfully!")
        
        return (dataset_name, stats, None)
        
    except Exception as e:
        logger.exception(f"Fatal error during {dataset_name} ingestion: {e}")
        return (dataset_name, None, e)


async def run_bootstrap() -> bool:
    """
    Run the bootstrap grading scales step.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("Step 1: Bootstrapping grading scales")
    logger.info("=" * 80)
    
    try:
        stats = await bootstrap_grading_scales()
        
        logger.info("=" * 80)
        logger.info("Bootstrap Summary:")
        logger.info(f"  Total mappings: {stats.get('total_mappings', 0)}")
        logger.info(f"  Exact: {stats.get('exact_mappings', 0)}")
        logger.info(f"  Approximate: {stats.get('approximate_mappings', 0)}")
        logger.info(f"  Manual review: {stats.get('manual_review_required', 0)}")
        logger.info("=" * 80)
        logger.info("✅ Bootstrap completed successfully!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Bootstrap failed: {e}")
        return False


def save_statistics_to_csv(
    ingestion_results: Dict[str, Tuple[Optional[OperationStatistics], Optional[Exception]]]
) -> Path:
    """
    Save all ingestion statistics to a CSV file.
    
    Args:
        ingestion_results: Dictionary mapping dataset names to (stats, error) tuples
        
    Returns:
        Path to the created CSV file
    """
    # Create output directory if it doesn't exist
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"ingestion_statistics_{timestamp}.csv"
    
    # Collect all unique keys from item_counts and error_counts for column headers
    all_item_types = set()
    all_error_types = set()
    
    for dataset_name, (stats, error) in ingestion_results.items():
        if stats is not None:
            all_item_types.update(stats.item_counts.keys())
            all_error_types.update(stats.error_counts.keys())
    
    # Sort for consistent column order
    item_type_columns = sorted(all_item_types)
    error_type_columns = sorted(all_error_types)
    
    # Define CSV columns
    base_columns = [
        "dataset_name",
        "status",
        "total_items",
        "processed_items",
        "successful_items",
        "failed_items",
        "skipped_items",
        "error_count",
    ]
    
    # Add dynamic columns for item_counts
    item_count_columns = [f"item_count_{item_type}" for item_type in item_type_columns]
    
    # Add dynamic columns for error_counts
    error_count_columns = [f"error_count_{error_type}" for error_type in error_type_columns]
    
    # Combine all columns
    fieldnames = base_columns + item_count_columns + error_count_columns + ["item_counts_json", "error_counts_json", "error_message"]
    
    # Write CSV file
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for dataset_name, (stats, error) in ingestion_results.items():
            row: dict[str, Any] = {"dataset_name": dataset_name}
            
            if error is not None:
                row["status"] = "fatal_error"
                row["error_message"] = str(error)
                # Fill rest with empty/default values
                for col in base_columns[2:]:  # Skip dataset_name and status
                    row[col] = 0
                for col in item_count_columns + error_count_columns:
                    row[col] = 0
                row["item_counts_json"] = ""
                row["error_counts_json"] = ""
            elif stats is None:
                row["status"] = "no_stats"
                row["error_message"] = "No statistics returned"
                for col in base_columns[2:]:
                    row[col] = 0
                for col in item_count_columns + error_count_columns:
                    row[col] = 0
                row["item_counts_json"] = ""
                row["error_counts_json"] = ""
            else:
                row["status"] = "success" if stats.failed_items == 0 else "completed_with_errors"
                row["total_items"] = stats.total_items
                row["processed_items"] = stats.processed_items
                row["successful_items"] = stats.successful_items
                row["failed_items"] = stats.failed_items
                row["skipped_items"] = stats.skipped_items
                row["error_count"] = len(stats.errors)
                
                # Fill item_counts columns
                for item_type in item_type_columns:
                    row[f"item_count_{item_type}"] = stats.item_counts.get(item_type, 0)
                
                # Fill error_counts columns
                for error_type in error_type_columns:
                    row[f"error_count_{error_type}"] = stats.error_counts.get(error_type, 0)
                
                # Add JSON representations for full detail
                row["item_counts_json"] = json.dumps(dict(stats.item_counts), sort_keys=True)
                row["error_counts_json"] = json.dumps(dict(stats.error_counts), sort_keys=True)
                row["error_message"] = ""
            
            writer.writerow(row)
    
    return csv_path


async def run_all_ingestions() -> Dict[str, Tuple[Optional[OperationStatistics], Optional[Exception]]]:
    """
    Run all ingest scripts concurrently using asyncio.gather() with a semaphore
    to limit concurrent database connections and avoid pool exhaustion.
    All ingests run on the main event loop so they share the same connection pool.
    
    Returns:
        Dictionary mapping dataset names to (stats, error) tuples
    """
    logger.info("=" * 80)
    logger.info("Step 2: Running all ingest scripts concurrently")
    logger.info(f"Total datasets: {len(INGEST_FUNCTIONS)}")
    logger.info("Note: Using semaphore to limit concurrent ingestions and avoid connection pool exhaustion")
    logger.info("=" * 80)
    
    # Each running dataset internally spawns many concurrent coroutines, each
    # needing a DB connection. Dividing by 8 gives each active dataset ~8
    # connections on average before the pool saturates.
    max_concurrent = max(1, db_config.max_connections // 8)
    semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Limiting concurrent ingestions to {max_concurrent} (pool max: {db_config.max_connections})")

    # Run-level timing + a running done/total tally with a rolling ETA, so a 30-90 min
    # full setup reports progress instead of going silent.
    total = len(INGEST_FUNCTIONS)
    run_start = time.monotonic()
    progress = {"done": 0}
    timings: Dict[str, float] = {}

    # Wrapper to limit concurrency with semaphore (all ingests run on main event loop
    # so they share the same connection pool; running any ingest in another thread's
    # event loop would cause "Lock is bound to a different event loop" errors)
    async def ingest_with_semaphore(dataset_name: str, ingest_func):
        """Run ingestion with semaphore to limit concurrent database connections."""
        async with semaphore:
            t0 = time.monotonic()
            result = await ingest_with_summary(dataset_name, ingest_func)
            elapsed = time.monotonic() - t0
            timings[dataset_name] = elapsed
            progress["done"] += 1
            done = progress["done"]
            avg = (time.monotonic() - run_start) / done
            eta = avg * (total - done)
            logger.info(
                f"[{done}/{total}] {dataset_name} done ({elapsed:.1f}s) — "
                f"{total - done} remaining, ~{eta:.0f}s left"
            )
            return result
    
    # Create tasks for all ingest functions (all on main loop, same pool)
    tasks = [
        ingest_with_semaphore(dataset_name, ingest_func)
        for dataset_name, ingest_func in INGEST_FUNCTIONS
    ]
    
    # Run all tasks concurrently (semaphore will limit actual concurrency)
    # Use return_exceptions=True to capture exceptions as results and continue processing
    # This allows other datasets to complete even if one fails
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert results to dictionary
    # Handle both normal results (tuple) and exceptions
    results_dict = {}
    for i, result in enumerate(results):
        dataset_name = INGEST_FUNCTIONS[i][0]
        if isinstance(result, Exception):
            # Exception occurred - wrap it
            logger.error(f"Fatal error in {dataset_name}: {result}", exc_info=result)
            results_dict[dataset_name] = (None, result)
        elif isinstance(result, tuple) and len(result) == 3:
            # Normal result: (dataset_name, stats, error)
            results_dict[dataset_name] = (result[1], result[2])
        else:
            # Unexpected format
            logger.warning(f"Unexpected result format for {dataset_name}: {type(result)}")
            results_dict[dataset_name] = (None, None)

    # Run-level timing summary
    total_elapsed = time.monotonic() - run_start
    if timings:
        avg = sum(timings.values()) / len(timings)
        fastest = min(timings.items(), key=lambda kv: kv[1])
        slowest = max(timings.items(), key=lambda kv: kv[1])
        h, rem = divmod(int(total_elapsed), 3600)
        m, s = divmod(rem, 60)
        logger.info("=" * 80)
        logger.info("Ingestion timing summary")
        logger.info(f"  Total wall-clock time : {h:02d}:{m:02d}:{s:02d}")
        logger.info(f"  Avg time per dataset  : {avg:.1f}s")
        logger.info(f"  Fastest               : {fastest[0]} ({fastest[1]:.1f}s)")
        logger.info(f"  Slowest               : {slowest[0]} ({slowest[1]:.1f}s)")
        logger.info("=" * 80)

    return results_dict

async def main() -> int:
    """Main entry point for the setup script."""
    parser = argparse.ArgumentParser(
        description="Set up the full database by bootstrapping scales and ingesting all datasets"
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip the bootstrap step (useful if already run)",
    )
    args = parser.parse_args()
    
    # Initialize connection pool
    try:
        await init_pool()
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        return 1
    
    try:
        # Step 1: Bootstrap (unless skipped)
        bootstrap_success = True
        if not args.skip_bootstrap:
            bootstrap_success = await run_bootstrap()
            if not bootstrap_success:
                logger.warning("Bootstrap failed, but continuing with ingestions...")
        else:
            logger.info("Skipping bootstrap step (--skip-bootstrap flag set)")
        
        # Step 2: Run all ingestions concurrently
        ingestion_results = await run_all_ingestions()
        
        # Step 3: Run validation for all successfully ingested datasets
        logger.info("")
        logger.info("=" * 80)
        logger.info("Step 3: Running validation checks")
        logger.info("=" * 80)

        # Run global FK checks once (annotation → image and patient_images cannot be
        # dataset-scoped, so running them once here avoids repeating the same full-table
        # scans 39 times in the per-dataset loop below)
        logger.info("Running global foreign key checks...")
        global_fk_violations = await validate_all_foreign_keys(dataset_id=None)
        if global_fk_violations:
            logger.warning(f"Global FK violations found: {len(global_fk_violations)}")
            for v in global_fk_violations:
                logger.warning(f"  {v.message}")
        else:
            logger.info("Global FK checks passed")

        validation_results = {}
        for dataset_name, (stats, error) in ingestion_results.items():
            if error is not None:
                logger.warning(f"Skipping validation for {dataset_name} (ingestion failed)")
                validation_results[dataset_name] = None
                continue

            try:
                dataset_id = generate_dataset_uuid(dataset_name)
                logger.info(f"Validating {dataset_name} (id={dataset_id})...")

                # Get actual DB counts for this dataset
                db_stats = await get_dataset_stats(dataset_id)
                logger.info(f"  DB counts for {dataset_name}:")
                for table, count in sorted(db_stats.items()):
                    if count > 0:
                        logger.info(f"    {table}: {count}")

                # skip_global_fk_checks=True: global FK checks already ran above
                report = await validate_dataset(
                    dataset_id=dataset_id,
                    check_orphans=True,
                    check_foreign_keys=True,
                    skip_global_fk_checks=True,
                )
                validation_results[dataset_name] = report

                # Log the full structured summary from the report
                for line in report.summary().splitlines():
                    if not report.is_valid:
                        logger.warning(line)
                    else:
                        logger.info(line)
            except Exception as e:
                logger.error(f"Validation failed for {dataset_name}: {e}", exc_info=e)
                validation_results[dataset_name] = None

        logger.info("")
        logger.info("Validation Summary:")
        valid_count = sum(1 for r in validation_results.values() if r and r.is_valid)
        invalid_count = sum(1 for r in validation_results.values() if r and not r.is_valid)
        skipped_count = sum(1 for r in validation_results.values() if r is None)
        logger.info(f"  ✅ Valid: {valid_count}")
        logger.info(f"  ⚠️  Issues found: {invalid_count}")
        logger.info(f"  ⏭️  Skipped: {skipped_count}")
        
        # Step 4: Save statistics to CSV
        csv_path = save_statistics_to_csv(ingestion_results)
        logger.info(f"Statistics saved to: {csv_path}")

        # Step 5: Generate final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 80)
        
        successful_datasets = []
        failed_datasets = []
        error_datasets = []
        
        total_items = 0
        total_successful = 0
        total_failed = 0
        total_skipped = 0
        
        for dataset_name, (stats, error) in ingestion_results.items():
            if error is not None:
                error_datasets.append(dataset_name)
            elif stats is None:
                failed_datasets.append(dataset_name)
            elif stats.failed_items > 0:
                failed_datasets.append(dataset_name)
                total_items += stats.total_items
                total_successful += stats.successful_items
                total_failed += stats.failed_items
                total_skipped += stats.skipped_items
            else:
                successful_datasets.append(dataset_name)
                total_items += stats.total_items
                total_successful += stats.successful_items
                total_failed += stats.failed_items
                total_skipped += stats.skipped_items
        
        logger.info(f"Bootstrap: {'✅ Success' if bootstrap_success else '❌ Failed'}")
        logger.info("")
        logger.info(f"Datasets processed: {len(ingestion_results)}")
        logger.info(f"  ✅ Successful: {len(successful_datasets)}")
        logger.info(f"  ⚠️  Completed with errors: {len(failed_datasets)}")
        logger.info(f"  ❌ Fatal errors: {len(error_datasets)}")
        
        if successful_datasets:
            logger.info("")
            logger.info("Successful datasets:")
            for name in sorted(successful_datasets):
                logger.info(f"  ✅ {name}")
        
        if failed_datasets:
            logger.info("")
            logger.warning("Datasets with errors:")
            for name in sorted(failed_datasets):
                logger.warning(f"  ⚠️  {name}")
        
        if error_datasets:
            logger.info("")
            logger.error("Datasets with fatal errors:")
            for name in sorted(error_datasets):
                logger.error(f"  ❌ {name}")
        
        logger.info("")
        logger.info("Overall Statistics:")
        logger.info(f"  Total items: {total_items}")
        logger.info(f"  Successful: {total_successful}")
        logger.info(f"  Failed: {total_failed}")
        logger.info(f"  Skipped: {total_skipped}")
        logger.info("=" * 80)

        # Step 6: Run IQA scoring + ROI detection on all ingested images
        logger.info("")
        logger.info("=" * 80)
        logger.info("Step 6: Running IQA scoring and ROI detection on all images")
        logger.info("=" * 80)
        try:
            await run_roi_iqa()
            logger.info("IQA + ROI detection completed successfully")
        except Exception as e:
            logger.error(f"IQA + ROI detection failed: {e}", exc_info=e)
            logger.warning("Continuing — annotations can be regenerated with: uv run python chaksudb/ingest/scripts/run_roi_iqa.py")

        # Step 7: Reconcile grade-conversion audit trail. The pm2-managed listener
        # (setup.sh) records these in real time; this sweep guarantees completeness for
        # any conversions that occurred while no listener was connected. Idempotent.
        logger.info("")
        logger.info("=" * 80)
        logger.info("Step 7: Reconciling grade-conversion provenance audit trail")
        logger.info("=" * 80)
        try:
            processed = await reconcile_grade_conversions()
            logger.info(f"Reconciled grade-conversion audit rows ({processed} gradings)")
        except Exception as e:
            logger.error(f"Grade-conversion reconciliation failed: {e}", exc_info=e)
            logger.warning("Continuing — rerun with: uv run python -c 'import asyncio; from chaksudb.ingest.framework.provenance import reconcile_grade_conversions; asyncio.run(reconcile_grade_conversions())'")

        # Step 8: Bootstrap the expert dictionary keyword vocabulary. Run last so it
        # seeds keyword_vocabulary from chaksudb/common/dictionary.py once everything
        # else is in place. Idempotent.
        logger.info("")
        logger.info("=" * 80)
        logger.info("Step 8: Bootstrapping expert dictionary keyword vocabulary")
        logger.info("=" * 80)
        try:
            vocab_stats = await bootstrap_dictionary_vocab()
            logger.info(
                "Dictionary vocab bootstrapped: %d condition terms, %d synonym terms",
                vocab_stats.get("condition_terms", 0),
                vocab_stats.get("synonym_terms", 0),
            )
        except Exception as e:
            logger.error(f"Dictionary vocab bootstrap failed: {e}", exc_info=e)
            logger.warning("Continuing — rerun with: uv run python scripts/bootstrap_dictionary_vocab.py")

        # Determine exit code
        if error_datasets or (not bootstrap_success and not args.skip_bootstrap):
            logger.error("Setup completed with fatal errors")
            return 1
        elif failed_datasets:
            logger.warning("Setup completed with some errors")
            return 1
        else:
            logger.info("✅ Setup completed successfully!")
            return 0
            
    except Exception as e:
        logger.exception(f"Fatal error during setup: {e}")
        return 1
    finally:
        # Close connection pool
        await close_pool()
        logger.info("Database connection pool closed")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
