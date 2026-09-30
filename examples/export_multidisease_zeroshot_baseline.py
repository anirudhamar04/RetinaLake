#!/usr/bin/env -S uv run
"""
Export OOD test-only fundus datasets to a single Parquet for zero-shot evaluation.

Run from repo root:  uv run examples/export_multidisease_zeroshot_baseline.py

Datasets: DeepDRiD, SUSTech-SYSU, JICHI, BRSET, PAPILA, G1020, AIROGS (justRAIGS)

Final parquet columns:
  image_id, file_path, resolution_width, resolution_height, dataset_name, zeroshot_label

Note: MMAC is not currently ingested in the database.

Export strategy:
  - DR datasets: grading annotation (disease_type="DR", dr_grade 0-4)
  - G1020, AIROGS: classification binary (class_name="glaucoma", task_type="binary")
  - PAPILA: classification multi-label (class_name="glaucoma", task_type="multi_label",
            keys: normal / glaucoma / glaucoma_suspicious)
  The three exports are merged into one raw parquet before label resolution.
"""

from pathlib import Path

import pandas as pd

from chaksudb.export import ExportSpec, export

OUT_DIR = Path("examples/export_output/zeroshot_baseline")

# ── Zero-shot label texts per dataset ────────────────────────────────────────
DR_TEXTS = [
    "normal",
    "mild diabetic retinopathy",
    "moderate diabetic retinopathy",
    "severe diabetic retinopathy",
    "proliferative diabetic retinopathy",
]

ZEROSHOT_TEXTS = {
    "DeepDRiD": DR_TEXTS,
    "SUSTech-SYSU": DR_TEXTS,
    "JICHI": DR_TEXTS,
    "BRSET": DR_TEXTS,
    "PAPILA": ["normal", "glaucoma", "suspected glaucoma"],
    "G1020": ["normal", "glaucoma"],
    "AIROGS": ["normal", "glaucoma"],
}


def _export_dr_datasets() -> pd.DataFrame:
    """Export DR datasets via grading annotation."""
    spec = ExportSpec(
        dataset_names=["DeepDRiD", "SUSTech-SYSU", "JICHI", "BRSET"],
        modalities=["fundus"],
        annotation_tasks=["grading"],
        disease_types=["DR"],
        annotation_source="prefer_consensus",
        include_original_grade=True,
        include_scaled_grade=True,
        require_annotations_mode="all",
        caption_mode="all",
    )
    path = OUT_DIR / "_raw_dr.parquet"
    export(spec, parquet_path=path)
    return pd.read_parquet(path)


def _export_binary_glaucoma_datasets() -> pd.DataFrame:
    """Export G1020 and AIROGS via binary glaucoma classification."""
    spec = ExportSpec(
        dataset_names=["G1020", "AIROGS"],
        modalities=["fundus"],
        annotation_tasks=["classification"],
        classification_class_names=["glaucoma"],
        classification_task_types={"glaucoma": "binary"},
        classification_label_type="int",
        annotation_source="prefer_consensus",
        require_annotations_mode="all",
        caption_mode="all",
    )
    path = OUT_DIR / "_raw_glaucoma_binary.parquet"
    export(spec, parquet_path=path)
    return pd.read_parquet(path)


def _export_papila() -> pd.DataFrame:
    """Export PAPILA via multi-label glaucoma classification."""
    spec = ExportSpec(
        dataset_names=["PAPILA"],
        modalities=["fundus"],
        annotation_tasks=["classification"],
        classification_class_names=["glaucoma"],
        classification_task_types={"glaucoma": "multi_label"},
        multi_label_keys={"glaucoma": ["normal", "glaucoma", "glaucoma_suspicious"]},
        classification_label_type="int",
        annotation_source="prefer_consensus",
        require_annotations_mode="all",
        caption_mode="all",
    )
    path = OUT_DIR / "_raw_papila.parquet"
    export(spec, parquet_path=path)
    return pd.read_parquet(path)


def export_ood_baseline() -> Path:
    """Run three sub-exports and merge into a single raw parquet."""
    df_dr = _export_dr_datasets()
    df_binary = _export_binary_glaucoma_datasets()
    df_papila = _export_papila()

    df = pd.concat([df_dr, df_binary, df_papila], ignore_index=True)

    path = OUT_DIR / "ood_zeroshot_raw.parquet"
    df.to_parquet(path, index=False)
    return path


# ── Label resolvers ───────────────────────────────────────────────────────────

def _resolve_dr_label(row: pd.Series) -> str:
    """Map dr_grade (0-4) to text label."""
    grade = row.get("dr_grade")
    if pd.isna(grade):
        return None
    return DR_TEXTS[int(grade)]


def _resolve_papila_label(row: pd.Series) -> str:
    """Map PAPILA glaucoma multi-label flags to text."""
    if row.get("glaucoma_glaucoma") == 1:
        return "glaucoma"
    if row.get("glaucoma_glaucoma_suspicious") == 1:
        return "suspected glaucoma"
    if row.get("glaucoma_normal") == 1:
        return "normal"
    return None


def _resolve_binary_glaucoma_label(row: pd.Series) -> str:
    """Map binary glaucoma classification (0/1) to text."""
    val = row.get("glaucoma_label")
    if pd.isna(val):
        return None
    return "glaucoma" if int(val) == 1 else "normal"


LABEL_RESOLVERS = {
    "DeepDRiD": _resolve_dr_label,
    "SUSTech-SYSU": _resolve_dr_label,
    "JICHI": _resolve_dr_label,
    "BRSET": _resolve_dr_label,
    "PAPILA": _resolve_papila_label,
    "G1020": _resolve_binary_glaucoma_label,
    "AIROGS": _resolve_binary_glaucoma_label,
}


def build_zeroshot_parquet(raw_path: Path) -> Path:
    """
    Read the raw export, resolve each row to its zeroshot_label text,
    and write a clean parquet with only the needed columns.
    """
    df = pd.read_parquet(raw_path)

    def resolve_label(row):
        resolver = LABEL_RESOLVERS.get(row["dataset_name"])
        if resolver is None:
            return None
        return resolver(row)

    df["zeroshot_label"] = df.apply(resolve_label, axis=1)

    before = len(df)
    df = df.dropna(subset=["zeroshot_label"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with no resolvable label")

    df = df[["image_id", "file_path", "resolution_width", "resolution_height",
             "dataset_name", "zeroshot_label"]]

    out_path = OUT_DIR / "ood_zeroshot_baseline.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("OOD Zero-Shot Baseline Export (fundus, single parquet)")
    print("=" * 55)

    raw = export_ood_baseline()
    print(f"  Raw export -> {raw}")

    final = build_zeroshot_parquet(raw)
    print(f"  Final       -> {final}")

    df = pd.read_parquet(final)
    print(f"\n  Total rows: {len(df)}")
    print(f"\n  Per-dataset breakdown:")
    for ds_name, group in df.groupby("dataset_name"):
        counts = group["zeroshot_label"].value_counts().to_dict()
        print(f"    {ds_name:15s} ({len(group):5d} images) -> {counts}")

    print(f"\n  Zero-shot prompt texts per dataset:")
    for ds, texts in ZEROSHOT_TEXTS.items():
        print(f"    {ds:15s} -> {texts}")

    print(f"\n  Note: MMAC is not currently ingested in the database.")


if __name__ == "__main__":
    main()
