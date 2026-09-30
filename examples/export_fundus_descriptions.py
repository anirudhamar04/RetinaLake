"""
Export all fundus images that have clinical descriptions to separate Parquet files
for train, test, and val splits.

- test  : images assigned to the 'test' split
- val   : images assigned to the 'val' split
- train : images assigned to the 'train' split OR from datasets with no split assignments

Usage:
    uv run python examples/export_fundus_descriptions.py
    uv run python examples/export_fundus_descriptions.py --output data/exports/fundus_desc/
    uv run python examples/export_fundus_descriptions.py --datasets EYEPACS IDRID
"""

import argparse
import logging
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chaksudb.export import ExportSpec, export

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fundus images with descriptions split into train/test/val"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/export_output/"),
        help="Output directory for train/test/val Parquet files (default: examples/export_output/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Restrict to specific dataset names (default: all datasets)",
    )
    return parser.parse_args()


def _log_summary(label: str, path: Path) -> None:
    table = pq.read_table(path)
    logger.info(f"\n--- {label} ({table.num_rows} rows) ---")
    logger.info(f"  File   : {path}")
    if table.num_rows > 0:
        df = table.to_pandas()
        if "dataset_name" in df.columns:
            logger.info(f"  Images per dataset:\n{df['dataset_name'].value_counts().to_string()}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base_spec = ExportSpec(
        dataset_names=args.datasets,
        modalities=["fundus"],
        caption_mode="all",
        require_annotations_mode="none",
    )

    test_path = args.output / "fundus_descriptions_test.parquet"
    val_path = args.output / "fundus_descriptions_val.parquet"
    train_path = args.output / "fundus_descriptions_train.parquet"

    # --- test ---
    logger.info("Exporting test split...")
    export(
        base_spec.model_copy(update={"split_names": ["test"]}),
        parquet_path=test_path,
    )

    # --- val ---
    logger.info("Exporting val split...")
    export(
        base_spec.model_copy(update={"split_names": ["val"]}),
        parquet_path=val_path,
    )

    # --- train = explicit 'train' split + all images with no split assignment ---
    # Export everything (no split filter), then subtract test and val image_ids.
    logger.info("Exporting all images (to derive train)...")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, dir=args.output) as tmp:
        tmp_path = Path(tmp.name)

    try:
        export(base_spec, parquet_path=tmp_path)

        test_ids = set(pq.read_table(test_path, columns=["image_id"]).column("image_id").to_pylist())
        val_ids = set(pq.read_table(val_path, columns=["image_id"]).column("image_id").to_pylist())
        excluded = test_ids | val_ids

        all_table = pq.read_table(tmp_path)
        all_df = all_table.to_pandas()
        train_df = all_df[~all_df["image_id"].isin(excluded)].reset_index(drop=True)

        pq.write_table(pa.Table.from_pandas(train_df, schema=all_table.schema), train_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # --- summaries ---
    logger.info("\n=== Export complete ===")
    _log_summary("TRAIN", train_path)
    _log_summary("TEST", test_path)
    _log_summary("VAL", val_path)


if __name__ == "__main__":
    main()
