"""
Concept-centric classification export.

Demonstrates the ML-concept interface: a concept (e.g. "glaucoma") is retrievable as a
unified binary column regardless of how each dataset stored it — AIROGS stored it as a
binary task, PAPILA as a multi_label sub-key, FIVES as the winning class of a multi_class
task. No need to know each dataset's class vocabulary up front.

    uv run python examples/export_classification_concepts.py
"""

import asyncio

from chaksudb.db.connection import init_pool
from chaksudb.db.queries.annotation_types import list_classification_tasks
from chaksudb.export.api import export
from chaksudb.export.spec import ExportSpec


async def main() -> None:
    await init_pool()

    # 1. Discover what classification tasks exist (no hardcoding the vocabulary).
    tasks = await list_classification_tasks()
    print("Classification tasks present:")
    for t in tasks:
        print(f"  task_name={t['task_name']:<22} type={t['task_type']:<12} "
              f"concepts={t['concepts']}")

    # 2. Export glaucoma + DR as unified per-concept binary columns, and keep only
    #    glaucoma-positive images — across binary / multi_label / multi_class storage.
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_concepts=["glaucoma", "DR"],     # -> glaucoma_present, DR_present (0/1)
        classification_positive_for=["glaucoma"],        # cross-shape row filter
    )
    # parquet_path is an argument to export(), NOT a field on ExportSpec.
    path = export(spec, parquet_path="examples/export_output/glaucoma_concept.parquet")
    print(f"\nWrote concept export to: {path}")


if __name__ == "__main__":
    asyncio.run(main())
