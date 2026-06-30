"""
Split assignment utilities for dataset ingestion.

This module provides utilities for:
- Registering dataset splits (train/val/test)
- Assigning images to splits
- Bulk assignment operations for efficient ingestion
- Auto stratified splitting for datasets with no/partial splits
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional, Sequence

from chaksudb.db.models import DatasetSplit, ImageSplit
from chaksudb.db.queries import upsert_dataset_split, upsert_image_split
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_split_uuid,
    generate_image_split_uuid,
)

logger = logging.getLogger(__name__)


async def register_dataset_split(
    dataset_id: uuid.UUID,
    split_name: str,
    split_type: str,
    task_type: Optional[str] = None,
    image_count: Optional[int] = None,
) -> uuid.UUID:
    """
    Register a dataset split in the database.

    Args:
        dataset_id: Dataset UUID
        split_name: Name of the split (e.g., 'train', 'val', 'test', 'validation')
        split_type: Type of split - must be one of:
            - 'explicit': Defined by folder structure or explicit files
            - 'metadata_defined': Defined in metadata (CSV column, JSON)
            - 'user_defined': Custom split defined by user
            - 'undefined': No split information available
        task_type: Optional task type if split is task-specific (e.g., 'grading', 'segmentation')
        image_count: Optional total number of images in this split

    Returns:
        split_id UUID of the registered split

    Example:
        ```python
        # Register a training split for a grading task
        split_id = await register_dataset_split(
            dataset_id=dataset_id,
            split_name="train",
            split_type="explicit",
            task_type="grading",
            image_count=1000
        )
        ```
    """
    # Generate deterministic UUID for the split
    split_id = generate_dataset_split_uuid(
        dataset_id=dataset_id,
        split_name=split_name,
        split_type=split_type,
        task_type=task_type,
    )

    # Create split model
    split = DatasetSplit(
        split_id=split_id,
        dataset_id=dataset_id,
        split_name=split_name,
        split_type=split_type,
        task_type=task_type,
        image_count=image_count,
        created_at=datetime.now(),
    )

    # Store in database (idempotent upsert)
    await upsert_dataset_split(split)

    logger.debug(
        f"Registered dataset split '{split_name}' ({split_type}) for dataset {dataset_id}"
        + (f" - task: {task_type}" if task_type else "")
    )

    return split_id


async def assign_image_to_split(
    image_id: uuid.UUID,
    split_id: uuid.UUID,
    task_type: Optional[str] = None,
    is_primary: bool = True,
) -> uuid.UUID:
    """
    Assign an image to a dataset split.

    Args:
        image_id: Image UUID
        split_id: Dataset split UUID
        task_type: Optional task type if assignment is task-specific
        is_primary: Whether this is the primary split assignment for this image

    Returns:
        assignment_id UUID of the assignment record

    Example:
        ```python
        # Assign an image to the training split
        assignment_id = await assign_image_to_split(
            image_id=image_id,
            split_id=train_split_id,
            task_type="grading",
            is_primary=True
        )
        ```
    """
    # Generate deterministic UUID for the assignment
    assignment_id = generate_image_split_uuid(
        image_id=image_id,
        split_id=split_id,
        task_type=task_type,
    )

    # Create image split model
    image_split = ImageSplit(
        assignment_id=assignment_id,
        image_id=image_id,
        split_id=split_id,
        task_type=task_type,
        is_primary=is_primary,
        created_at=datetime.now(),
    )

    # Store in database (idempotent upsert)
    await upsert_image_split(image_split)

    logger.debug(
        f"Assigned image {image_id} to split {split_id}"
        + (f" - task: {task_type}" if task_type else "")
    )

    return assignment_id


async def bulk_assign_images_to_split(
    image_ids: Sequence[uuid.UUID],
    split_id: uuid.UUID,
    task_type: Optional[str] = None,
    is_primary: bool = True,
) -> int:
    """
    Bulk assign multiple images to a dataset split.

    This is more efficient than calling assign_image_to_split in a loop
    when you have many images to assign.

    Args:
        image_ids: Sequence of image UUIDs to assign
        split_id: Dataset split UUID
        task_type: Optional task type if assignment is task-specific
        is_primary: Whether this is the primary split assignment for these images

    Returns:
        Number of images assigned

    Example:
        ```python
        # Bulk assign 1000 images to training split
        count = await bulk_assign_images_to_split(
            image_ids=train_image_ids,
            split_id=train_split_id,
            task_type="grading",
            is_primary=True
        )
        print(f"Assigned {count} images to training split")
        ```
    """
    count = 0
    for image_id in image_ids:
        await assign_image_to_split(
            image_id=image_id,
            split_id=split_id,
            task_type=task_type,
            is_primary=is_primary,
        )
        count += 1

    logger.debug(
        f"Bulk assigned {count} images to split {split_id}"
        + (f" - task: {task_type}" if task_type else "")
    )

    return count


async def register_standard_splits(
    dataset_id: uuid.UUID,
    split_type: str = "explicit",
    task_type: Optional[str] = None,
    train_count: Optional[int] = None,
    val_count: Optional[int] = None,
    test_count: Optional[int] = None,
) -> dict[str, uuid.UUID]:
    """
    Register standard train/val/test splits for a dataset.

    This is a convenience function for the common case of having
    train, validation, and test splits.

    Args:
        dataset_id: Dataset UUID
        split_type: Type of split (default: 'explicit')
        task_type: Optional task type if splits are task-specific
        train_count: Optional number of images in training split
        val_count: Optional number of images in validation split
        test_count: Optional number of images in test split

    Returns:
        Dictionary mapping split names to their UUIDs:
        {"train": train_split_id, "val": val_split_id, "test": test_split_id}

    Example:
        ```python
        # Register standard splits for a grading task
        splits = await register_standard_splits(
            dataset_id=dataset_id,
            split_type="explicit",
            task_type="grading",
            train_count=800,
            val_count=100,
            test_count=100
        )
        train_split_id = splits["train"]
        ```
    """
    splits = {}

    # Register training split
    splits["train"] = await register_dataset_split(
        dataset_id=dataset_id,
        split_name="train",
        split_type=split_type,
        task_type=task_type,
        image_count=train_count,
    )

    # Register validation split
    splits["val"] = await register_dataset_split(
        dataset_id=dataset_id,
        split_name="val",
        split_type=split_type,
        task_type=task_type,
        image_count=val_count,
    )

    # Register test split
    splits["test"] = await register_dataset_split(
        dataset_id=dataset_id,
        split_name="test",
        split_type=split_type,
        task_type=task_type,
        image_count=test_count,
    )

    logger.debug(
        f"Registered standard splits for dataset {dataset_id}"
        + (f" - task: {task_type}" if task_type else "")
    )

    return splits


async def assign_images_by_split_dict(
    split_assignments: dict[str, Sequence[uuid.UUID]],
    split_ids: dict[str, uuid.UUID],
    task_type: Optional[str] = None,
    is_primary: bool = True,
) -> dict[str, int]:
    """
    Assign images to splits based on a dictionary mapping.

    This is a convenience function for batch assigning images when you have
    them organized by split name.

    Args:
        split_assignments: Dictionary mapping split names to lists of image UUIDs
            e.g., {"train": [uuid1, uuid2, ...], "val": [uuid3, ...], "test": [uuid4, ...]}
        split_ids: Dictionary mapping split names to split UUIDs
            e.g., {"train": train_split_id, "val": val_split_id, "test": test_split_id}
        task_type: Optional task type if assignments are task-specific
        is_primary: Whether these are primary split assignments

    Returns:
        Dictionary mapping split names to number of images assigned
        e.g., {"train": 800, "val": 100, "test": 100}

    Example:
        ```python
        # First register splits
        splits = await register_standard_splits(dataset_id, split_type="explicit")
        
        # Then assign images
        assignments = {
            "train": train_image_ids,
            "val": val_image_ids,
            "test": test_image_ids
        }
        counts = await assign_images_by_split_dict(
            split_assignments=assignments,
            split_ids=splits,
            task_type="grading"
        )
        print(f"Assigned {counts['train']} training images")
        ```
    """
    counts = {}

    for split_name, image_ids in split_assignments.items():
        if split_name not in split_ids:
            logger.warning(
                f"Split name '{split_name}' not found in split_ids. Skipping."
            )
            continue

        split_id = split_ids[split_name]
        count = await bulk_assign_images_to_split(
            image_ids=image_ids,
            split_id=split_id,
            task_type=task_type,
            is_primary=is_primary,
        )
        counts[split_name] = count

    return counts


def _stratified_split(
    image_ids: list[uuid.UUID],
    labels: Optional[dict[uuid.UUID, Any]],
    test_size: float,
    random_seed: int,
) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    """Split image_ids into two groups using stratified sampling when labels are available.

    Falls back to random split if stratification is not possible (e.g. a class has
    too few samples to appear in both splits).
    """
    from sklearn.model_selection import train_test_split

    if len(image_ids) < 2:
        return image_ids, []

    # Determine minimum samples needed per class for stratification to work.
    # sklearn requires at least 2 samples per class (one in each side).
    stratify = None
    if labels:
        y = [str(labels.get(img_id, "__unlabeled__")) for img_id in image_ids]
        from collections import Counter
        if min(Counter(y).values()) >= 2:
            stratify = y

    try:
        a, b = train_test_split(
            image_ids,
            test_size=test_size,
            stratify=stratify,
            random_state=random_seed,
        )
    except ValueError:
        # Fallback: non-stratified random split
        a, b = train_test_split(
            image_ids,
            test_size=test_size,
            stratify=None,
            random_state=random_seed,
        )
    return list(a), list(b)


async def delete_splits_for_dataset(
    dataset_id: uuid.UUID,
    split_types: Optional[list[str]] = None,
) -> int:
    """
    Delete dataset_splits (and their image_splits assignments) for a dataset.

    Args:
        dataset_id: Dataset UUID.
        split_types: If given, only delete splits of these types
            (e.g. ['explicit', 'undefined']). If None, delete all split types.

    Returns:
        Number of dataset_splits rows deleted.
    """
    from chaksudb.db.connection import get_connection

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            if split_types:
                await cur.execute(
                    """
                    DELETE FROM image_splits
                    WHERE split_id IN (
                        SELECT split_id FROM dataset_splits
                        WHERE dataset_id = %s AND split_type = ANY(%s)
                    )
                    """,
                    (dataset_id, split_types),
                )
                await cur.execute(
                    """
                    DELETE FROM dataset_splits
                    WHERE dataset_id = %s AND split_type = ANY(%s)
                    RETURNING split_id
                    """,
                    (dataset_id, split_types),
                )
            else:
                await cur.execute(
                    """
                    DELETE FROM image_splits
                    WHERE split_id IN (
                        SELECT split_id FROM dataset_splits WHERE dataset_id = %s
                    )
                    """,
                    (dataset_id,),
                )
                await cur.execute(
                    "DELETE FROM dataset_splits WHERE dataset_id = %s RETURNING split_id",
                    (dataset_id,),
                )
            deleted = len(await cur.fetchall())
            await conn.commit()

    logger.debug(
        f"Deleted {deleted} split definition(s) for dataset {dataset_id}"
        + (f" (types: {split_types})" if split_types else " (all types)")
    )
    return deleted


async def auto_stratified_splits(
    dataset_id: uuid.UUID,
    split_assignments: dict[str, list[uuid.UUID]],
    labels: Optional[dict[uuid.UUID, Any]] = None,
    split_type: str = "user_defined",
    task_type: Optional[str] = None,
    random_seed: int = 42,
) -> tuple[dict[str, uuid.UUID], dict[str, int]]:
    """Register and assign dataset splits, auto-generating missing splits via stratified sampling.

    Three cases are handled automatically based on the keys present in *split_assignments*:

    1. **train / val / test** — All three splits already exist.  Register and assign as-is.
    2. **train / test** — Val split is missing.  The training images are split 90 / 10
       (train → train + val) using stratified sampling when *labels* are provided.
    3. **train only** (or empty) — No test or val splits exist.  The full image pool is
       first split 90 / 10 into train + test (stratified), then the train portion is
       further split 90 / 10 into train + val (stratified).

    Stratification is performed on the values in *labels* (a ``{image_id: class_label}``
    mapping).  When *labels* is ``None``, or when a class has fewer than 2 samples (making
    stratified splitting impossible), the function falls back to a seeded random split.

    Known (input) splits use the provided *split_type*.  Any split that is derived by
    this function (i.e. val in case 2, and all splits in case 3) is always stored with
    ``split_type="user_defined"`` to distinguish it from original dataset splits.

    Args:
        dataset_id: Dataset UUID.
        split_assignments: Mapping of known split names to their image UUID lists.
            Accepted key combinations: ``{"train", "val", "test"}``,
            ``{"train", "test"}``, ``{"train"}``, or ``{}``.
        labels: Optional ``{image_id: label}`` dict used for stratification.
            Labels can be any hashable type (int grade, string class name, etc.).
        split_type: Split type for *known* (input) splits (default ``"user_defined"``).
            Must be one of ``explicit``, ``metadata_defined``, ``user_defined``,
            ``undefined``.  Derived splits always use ``"user_defined"``.
        task_type: Optional task type string (e.g. ``"grading"``).
        random_seed: Random seed for reproducible splits (default 42).

    Returns:
        A ``(split_ids, counts)`` tuple where *split_ids* maps split names to their
        registered DB UUIDs and *counts* maps split names to the number of images
        assigned.
    """
    known_splits = set(split_assignments.keys())

    # Track which splits are derived so we can tag them "user_defined".
    derived_splits: set[str] = set()

    # ── Case 1: already have all three splits ──────────────────────────────────
    if {"train", "val", "test"} <= known_splits:
        final = {k: list(v) for k, v in split_assignments.items()}

    # ── Case 2: train + test, missing val ─────────────────────────────────────
    elif "train" in known_splits and "test" in known_splits:
        train_ids = list(split_assignments["train"])
        test_ids = list(split_assignments["test"])
        train_ids, val_ids = _stratified_split(
            train_ids, labels, test_size=0.1, random_seed=random_seed
        )
        final = {"train": train_ids, "val": val_ids, "test": test_ids}
        derived_splits = {"val"}

    # ── Case 3: train only (or empty) — derive all splits ─────────────────────
    else:
        all_ids = list(split_assignments.get("train", []))
        train_and_val_ids, test_ids = _stratified_split(
            all_ids, labels, test_size=0.1, random_seed=random_seed
        )
        train_ids, val_ids = _stratified_split(
            train_and_val_ids, labels, test_size=0.1, random_seed=random_seed
        )
        final = {"train": train_ids, "val": val_ids, "test": test_ids}
        derived_splits = {"train", "val", "test"}

    # ── Register splits in DB ─────────────────────────────────────────────────
    split_ids: dict[str, uuid.UUID] = {}
    for split_name, ids in final.items():
        effective_type = "user_defined" if split_name in derived_splits else split_type
        split_ids[split_name] = await register_dataset_split(
            dataset_id=dataset_id,
            split_name=split_name,
            split_type=effective_type,
            task_type=task_type,
            image_count=len(ids),
        )

    # ── Assign images ─────────────────────────────────────────────────────────
    counts: dict[str, int] = {}
    for split_name, ids in final.items():
        counts[split_name] = await bulk_assign_images_to_split(
            image_ids=ids,
            split_id=split_ids[split_name],
            task_type=task_type,
        )

    logger.info(
        f"auto_stratified_splits for dataset {dataset_id}: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
        + (f" [task={task_type}]" if task_type else "")
    )

    return split_ids, counts
