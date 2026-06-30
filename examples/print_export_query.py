#!/usr/bin/env -S uv run
"""
Print the SQL an ExportSpec generates — the single best way to understand the export layer.

Every export module (image, grading, classification, segmentation, …) contributes SELECT
columns, JOINs, and WHERE clauses. Printing the assembled SQL makes all of that concrete and is
the fastest way to debug "why is this column empty / this row missing?".

    uv run python examples/print_export_query.py

No database connection required — this only builds the query string.
"""

from chaksudb.export.spec import ExportSpec
from chaksudb.export.query_builder import QueryBuilder


def main() -> None:
    spec = ExportSpec(
        dataset_names=["IDRID"],
        annotation_tasks=["grading", "segmentation"],
        disease_types=["DR"],
        require_annotations_mode="all",
    )

    sql = QueryBuilder().build_query(spec).render_sql()

    # `sqlparse` (a dev dependency) pretty-prints it; fall back to raw SQL if absent.
    try:
        import sqlparse

        sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
    except ImportError:
        pass

    print(sql)


if __name__ == "__main__":
    main()
