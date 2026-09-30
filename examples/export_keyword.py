#!/usr/bin/env -S uv run
"""
Export clinical keywords and free-text descriptions.

The `keyword` and `description` tasks pull diagnostic keyword terms and free-form clinical notes
for datasets that have them (e.g. DeepEyeNet, 1000x39). For *generated* captions built from
structured annotations, see `export_fundus_descriptions.py`.

    uv run python examples/export_keyword.py
"""

from pathlib import Path

from chaksudb.export import ExportSpec, export

OUT_DIR = Path("examples/export_output")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = ExportSpec(
        annotation_tasks=["keyword", "description"],
        require_annotations_mode="any",   # keep images that have keywords OR a description
    )
    path = export(spec, parquet_path=OUT_DIR / "keyword_description.parquet")
    print(f"Keyword + description export: {path}")


if __name__ == "__main__":
    main()
