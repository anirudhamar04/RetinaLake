"""
Export pipeline: query the DB with ExportSpec, then export to Parquet or PyTorch.

Public API:
  - ExportSpec: define what to take from the DB (datasets, annotation tasks, filters, etc.)
  - export(spec, parquet_path=..., torch=..., spatial=..., transform=...): single entry point
  - Transform classes from chaksudb.export.transforms

Example:
  from chaksudb.export import ExportSpec, export
  from chaksudb.export.transforms import Resize, RandomHorizontalFlip, CLAHE, Normalize

  spec = ExportSpec(dataset_names=["EYEPACS"], annotation_tasks=["grading"], disease_types=["DR"])
  export(spec, parquet_path=Path("data.parquet"))
  dataloader = export(spec, torch="dataloader", spatial=[Resize(224)], batch_size=32, shuffle=True)
"""

from chaksudb.export.api import export
from chaksudb.export.spec import ExportSpec

__version__ = "2.0.0"

__all__ = [
    "ExportSpec",
    "export",
]
