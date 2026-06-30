"""
Dataset-faithful flat export — everything a single dataset has, flattened.

`build_dataset_spec(["BRSET"])` introspects the dataset and flattens all its labels into
columns that mirror the source CSV (disease panel -> one column per disease, the ICDR DR
grade, every quality parameter, anatomical findings, patient demographics). `["FIVES"]`
comes out with its disease multi_class columns plus its segmentation masks.

Handy when you want to train/test on a single dataset and just need all its labels.

    uv run python examples/export_dataset_flat.py BRSET
    uv run python examples/export_dataset_flat.py FIVES
"""

import asyncio
import sys

from chaksudb.db.connection import init_pool
from chaksudb.export.api import export
from chaksudb.export.discovery import build_dataset_spec


async def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "BRSET"
    await init_pool()

    # Everything the dataset has, flattened. Override anything you like, e.g.
    # build_dataset_spec([dataset], split_names=["test"]) to test on the test split only.
    spec = await build_dataset_spec([dataset])
    print(f"{dataset}: annotation_tasks={spec.annotation_tasks}")
    print(f"  disease_types={spec.disease_types}")
    print(f"  classification tasks={spec.classification_class_names}")
    print(f"  segmentation_types={spec.segmentation_types}")

    path = export(spec, parquet_path=f"examples/export_output/{dataset.lower()}_flat.parquet")
    print(f"Wrote flat export -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
