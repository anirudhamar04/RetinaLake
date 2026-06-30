"""
ContrastiveDataset: PyTorch Dataset for contrastive / retrieval training.

Returns (anchor, positive, negative) triplets mined from a Parquet file
based on a configurable label column.
"""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset

from chaksudb.export.path_resolution import resolve_local_path, resolve_paths_in_row
from chaksudb.export.torch_dataset import _pil_to_tensor

logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    """Dataset that yields (anchor, positive, negative) triplets.

    Pairs are mined based on a label column: images sharing the same label
    value are considered positives; images with different label values are
    negatives.

    Args:
        parquet_path: Path to the Parquet export file.
        label_column: Column name used to determine positive / negative pairs.
        transform: Optional image transform (PIL -> PIL or PIL -> Tensor).
        seed: Random seed for reproducible pair mining.
    """

    def __init__(
        self,
        parquet_path: Path,
        label_column: str,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.parquet_path = Path(parquet_path)
        self.label_column = label_column
        self.transform = transform
        self.rng = random.Random(seed)

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        import pyarrow.parquet as pq
        table = pq.read_table(self.parquet_path, columns=[label_column, "file_path"])
        self._rows = table.to_pylist()

        # Build label → row-index mapping
        self._label_to_indices: dict[Any, list[int]] = defaultdict(list)
        for idx, row in enumerate(self._rows):
            label = row.get(label_column)
            if label is not None:
                self._label_to_indices[label].append(idx)

        self._labels = list(self._label_to_indices.keys())
        # Pre-compute per-label negative label lists to avoid rebuilding in __getitem__
        self._neg_labels: dict[Any, list[Any]] = {
            lbl: [l for l in self._labels if l != lbl] for lbl in self._labels
        }
        logger.debug(
            "ContrastiveDataset: %d rows, %d unique labels",
            len(self._rows), len(self._labels),
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_row = resolve_paths_in_row(self._rows[idx])
        anchor_label = self._rows[idx].get(self.label_column)

        # Pick positive (same label, different index)
        pos_pool = self._label_to_indices.get(anchor_label, [idx])
        pos_idx = self.rng.choice(pos_pool)
        while pos_idx == idx and len(pos_pool) > 1:
            pos_idx = self.rng.choice(pos_pool)
        pos_row = resolve_paths_in_row(self._rows[pos_idx])

        # Pick negative (different label)
        neg_labels = self._neg_labels.get(anchor_label, [])
        if neg_labels:
            neg_label = self.rng.choice(neg_labels)
            neg_idx = self.rng.choice(self._label_to_indices[neg_label])
        else:
            neg_idx = idx
        neg_row = resolve_paths_in_row(self._rows[neg_idx])

        anchor_img = self._load_image(anchor_row)
        pos_img = self._load_image(pos_row)
        neg_img = self._load_image(neg_row)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        if isinstance(anchor_img, PILImage.Image):
            anchor_img = _pil_to_tensor(anchor_img)
            pos_img = _pil_to_tensor(pos_img)
            neg_img = _pil_to_tensor(neg_img)

        return anchor_img, pos_img, neg_img

    @staticmethod
    def _load_image(row: dict[str, Any]) -> PILImage.Image:
        fp = row.get("file_path")
        if not fp:
            raise ValueError("Row missing file_path")
        path = resolve_local_path(fp)
        return PILImage.open(path).convert("RGB")
