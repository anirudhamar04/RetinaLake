#!/usr/bin/env -S uv run
"""
Self-supervised (SSL) export: image-only parquet across many datasets.

`output_format="ssl"` produces a flat image table (no labels) suitable for self-supervised
pretraining. Combine with `include_fundus_roi=True` to carry the fundus circle for ROI masking.

    uv run python examples/ssl_export.py

Edit `DATASETS` for your corpus; outputs land in examples/export_output/.
"""

from pathlib import Path

from chaksudb.export import ExportSpec, export

OUT_DIR = Path("examples/export_output")

# Any datasets you have ingested. Trim this list to what's in your DB.
DATASETS = [
    "1000x39", "ACRIMA", "APTOS", "BRSET", "CHAKSU", "DDR", "DeepDRiD", "EYEPACS",
    "FIVES", "G1020", "IDRID", "MMAC", "ODIR-5K", "PAPILA", "REFUGE", "STARE",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = ExportSpec(
        dataset_names=DATASETS,
        split_names=["train", "val", "test"],
        modalities=["fundus"],
        include_fundus_roi=True,
        output_format="ssl",
    )
    path = export(spec, parquet_path=OUT_DIR / "ssl_pretrain.parquet")
    print(f"SSL pretraining export: {path}")


if __name__ == "__main__":
    main()
