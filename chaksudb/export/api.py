"""
Single entry point for export: DB -> query (ExportSpec) -> Parquet or PyTorch.

Use ExportSpec to define what to take from the DB, then call export() to write
Parquet and/or get a PyTorch Dataset or DataLoader with optional spatial and
photometric transforms.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union, Literal

from chaksudb.export.parquet_export import export_to_parquet_sync
from chaksudb.export.spec import ExportSpec

logger = logging.getLogger(__name__)
from chaksudb.export.torch_dataset import (
    ParquetDataset,
    QueryDataset,
    create_dataloader,
)


def export(
    spec: ExportSpec,
    *,
    parquet_path: Optional[Path] = None,
    coco_path: Optional[Path] = None,
    torch: Optional[Literal["dataset", "dataloader"]] = None,
    spatial: Optional[list] = None,
    transform: Any = None,
    collate_fn: Any = "default",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    parquet_batch_size: int = 5000,
    **dataloader_kwargs: Any,
) -> Union[Path, Any, None]:
    """
    Single entry point: export query results to Parquet and/or PyTorch.

    Args:
        spec: ExportSpec defining the query.
        parquet_path: If set, export rows to this Parquet file.
        coco_path: If set, write a COCO-format JSON sidecar alongside the
            Parquet export.  Requires ``parquet_path`` and localization
            annotations in the spec.
        torch: ``"dataset"`` or ``"dataloader"``.
        spatial: List of BaseSpatialTransform instances for the spatial pipeline.
        transform: Photometric transforms – a list of image-only transforms or a
            single callable ``PIL.Image -> PIL.Image``.  The old ``(image, annotations)``
            signature is no longer supported.
        collate_fn: ``"default"`` | ``"padded"`` | ``"packed"`` | callable.
        batch_size: Batch size for DataLoader (default 32).
        shuffle: Shuffle DataLoader (default False).
        num_workers: DataLoader workers (default 0).
        parquet_batch_size: Streaming batch size for Parquet export (default 5000).
        **dataloader_kwargs: Forwarded to DataLoader.

    Returns:
        Path if only Parquet was written; Dataset or DataLoader when ``torch`` is
        set; None if nothing was requested.
    """
    logger.info(
        "export called | datasets=%s tasks=%s mode=%s source=%s modalities=%s",
        spec.dataset_names or "all",
        spec.annotation_tasks or [],
        spec.require_annotations_mode,
        spec.annotation_source,
        spec.modalities or "all",
    )

    written_path: Optional[Path] = None

    if parquet_path is not None:
        logger.info("writing parquet -> %s (batch_size=%d)", parquet_path, parquet_batch_size)
        export_to_parquet_sync(spec, Path(parquet_path), batch_size=parquet_batch_size)
        written_path = Path(parquet_path)
        logger.info("parquet written: %s", written_path)

        # Write COCO JSON sidecar if requested
        if coco_path is not None:
            import pyarrow.parquet as pq
            from chaksudb.export.coco_export import export_coco_json

            table = pq.read_table(
                written_path,
                columns=["image_id", "file_path", "resolution_width", "resolution_height", "localization_annotations"],
            )
            rows = table.to_pylist()
            export_coco_json(
                rows,
                Path(coco_path),
                category_map=spec.detection_category_map,
            )

    if torch is None:
        return written_path

    if torch not in ("dataset", "dataloader"):
        raise ValueError(f'torch must be "dataset" or "dataloader", got {torch!r}')

    if written_path is None:
        raise ValueError(
            "parquet_path is required when torch='dataset' or torch='dataloader'. "
            "The DB is optimized for bulk streaming, not per-sample point lookups. "
            "Provide a parquet_path so rows are written once at full throughput, "
            "then __getitem__ reads from the file."
        )

    if torch == "dataset":
        return ParquetDataset(
            parquet_path=written_path,
            spatial=spatial,
            transform=transform,
            output_format=spec.output_format,
            class_names=spec.classification_class_names,
            disease_types=spec.disease_types,
        )

    if torch == "dataloader":
        return create_dataloader(
            parquet_path=written_path,
            spatial=spatial,
            transform=transform,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **dataloader_kwargs,
        )
