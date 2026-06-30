"""
Tests for the grade-conversion provenance audit path and the read/audit query layer.

Covers:
- record_grade_conversion idempotency (the core bug fix: re-running converges)
- the read helpers (get_chain, get_transformations_for_chain, get_lineage_for_image,
  find_orphan_transformations)
- reconcile_grade_conversions creating + idempotently re-recording audit rows, with the
  auto_convert_disease_grade trigger loaded.

All tests require a database (and, where noted, the grading trigger).
"""

import uuid
from datetime import datetime
from pathlib import Path

import psycopg
import pytest

from chaksudb.config.config import get_db_async_connection_string
from chaksudb.db.models import Dataset, DiseaseGrading, GradingScale, Image, ProvenanceChain
from chaksudb.db.queries import (
    find_orphan_transformations,
    get_chain,
    get_lineage_for_image,
    get_transformations_for_chain,
)
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.queries.grading import upsert_disease_grading, upsert_grading_scale
from chaksudb.db.queries.images import upsert_image
from chaksudb.db.queries.provenance import upsert_provenance_chain
from chaksudb.ingest.framework.provenance import reconcile_grade_conversions
from chaksudb.ingest.framework.provenance_listener import (
    TRANSFORMATION_TYPE,
    record_grade_conversion,
)

pytestmark = pytest.mark.requires_db


@pytest.fixture(scope="function")
async def grading_trigger(test_db_schema):
    """Load the auto_convert_disease_grade trigger for the duration of one test, then
    drop it.

    conftest applies only schema.sql, so trigger-dependent tests must load it explicitly.
    The trigger is dropped on teardown so it cannot leak into other tests in the shared
    session (it recomputes scaled_grade and would interfere with the Python-only scaling
    paths exercised elsewhere).
    """
    trigger_sql = (
        Path(__file__).parent.parent.parent
        / "schema" / "triggers" / "auto_convert_grading.sql"
    ).read_text()
    async with await psycopg.AsyncConnection.connect(get_db_async_connection_string()) as conn:
        await conn.execute(trigger_sql)
        await conn.commit()
    try:
        yield
    finally:
        async with await psycopg.AsyncConnection.connect(get_db_async_connection_string()) as conn:
            await conn.execute(
                "DROP TRIGGER IF EXISTS trigger_auto_convert_disease_grade ON disease_grading"
            )
            await conn.execute("DROP FUNCTION IF EXISTS auto_convert_disease_grade() CASCADE")
            await conn.commit()


def _event(chain_id: uuid.UUID, *, mode: str = "mapped") -> dict:
    """Build a NOTIFY-shaped grade_conversion event (string/int typed, like JSON)."""
    return {
        "mode": mode,
        "grading_id": str(uuid.UUID("11111111-1111-1111-1111-111111111111")),
        "image_id": str(uuid.UUID("22222222-2222-2222-2222-222222222222")),
        "scale_id": str(uuid.UUID("33333333-3333-3333-3333-333333333333")),
        "original_grade": "moderate",
        "disease_type": "DR",
        "scaled_grade": 2,
        "target_scale_id": str(uuid.UUID("44444444-4444-4444-4444-444444444444")),
        "target_scale_name": "ICDR_0_4",
        "provenance_chain_id": str(chain_id),
    }


async def _make_chain(chain_id: uuid.UUID, annotation_type: str = "grading") -> None:
    await upsert_provenance_chain(
        ProvenanceChain(
            chain_id=chain_id,
            unified_annotation_type=annotation_type,
            source_type="original",
            root_source_raw_data_id=None,
            source_annotation_ids=None,
            created_at=datetime.now(),
        )
    )


async def test_record_grade_conversion_is_idempotent(db_connection, test_db_schema):
    """Recording the same conversion event twice yields exactly one audit row + one link."""
    chain_id = uuid.UUID("a1a1a1a1-aaaa-4aaa-8aaa-aaaaaaaaaaa1")
    await _make_chain(chain_id)

    assert await record_grade_conversion(_event(chain_id)) is True
    assert await record_grade_conversion(_event(chain_id)) is True  # re-delivery converges

    transformations = await get_transformations_for_chain(chain_id)
    assert len(transformations) == 1
    assert transformations[0].operation_type == TRANSFORMATION_TYPE

    # Exactly one link row for this chain.
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM provenance_transformations WHERE chain_id = %s",
            (chain_id,),
        )
        assert (await cur.fetchone())[0] == 1


async def test_record_grade_conversion_skips_without_chain(db_connection, test_db_schema):
    """An event with no provenance_chain_id is skipped (returns False)."""
    event = _event(uuid.UUID("b2b2b2b2-bbbb-4bbb-8bbb-bbbbbbbbbbb2"))
    event["provenance_chain_id"] = None
    assert await record_grade_conversion(event) is False


async def test_recorded_transformation_is_not_orphan(db_connection, test_db_schema):
    """A linked transformation never appears in find_orphan_transformations."""
    chain_id = uuid.UUID("c3c3c3c3-cccc-4ccc-8ccc-ccccccccccc3")
    await _make_chain(chain_id)
    await record_grade_conversion(_event(chain_id))

    linked = await get_transformations_for_chain(chain_id)
    assert len(linked) == 1
    linked_id = linked[0].transformation_id

    orphan_ids = {t.transformation_id for t in await find_orphan_transformations()}
    assert linked_id not in orphan_ids


async def test_get_chain_roundtrip(db_connection, test_db_schema):
    chain_id = uuid.UUID("d4d4d4d4-dddd-4ddd-8ddd-ddddddddddd4")
    await _make_chain(chain_id, annotation_type="grading")
    chain = await get_chain(chain_id)
    assert chain is not None
    assert chain.chain_id == chain_id
    assert chain.unified_annotation_type == "grading"
    assert await get_chain(uuid.UUID("00000000-0000-4000-8000-000000000000")) is None


async def test_reconcile_grade_conversions_creates_and_is_idempotent(
    db_connection, grading_trigger
):
    """End-to-end: a converted grading row gets an audit row via reconciliation, idempotently.

    The trigger sets scaled_grade and emits NOTIFY, but no listener is running in the test,
    so the audit row only appears after reconcile_grade_conversions — and stays at one row
    when reconciliation runs again.
    """
    dataset_id = uuid.UUID("e5e5e5e5-eeee-4eee-8eee-eeeeeeeeeee5")
    image_id = uuid.UUID("e5e5e5e5-eeee-4eee-8eee-eeeeeeeeeee6")
    scale_id = uuid.UUID("e5e5e5e5-eeee-4eee-8eee-eeeeeeeeeee7")
    chain_id = uuid.UUID("e5e5e5e5-eeee-4eee-8eee-eeeeeeeeeee8")
    grading_id = uuid.UUID("e5e5e5e5-eeee-4eee-8eee-eeeeeeeeeee9")

    await upsert_dataset(Dataset(dataset_id=dataset_id, dataset_name="AUDIT_TEST"))
    await upsert_image(
        Image(image_id=image_id, dataset_id=dataset_id, file_path="audit/test.png")
    )
    # ICDR_0_4 is the canonical target scale; inserting on it makes the trigger set
    # scaled_grade = original_grade (same-scale branch).
    await upsert_grading_scale(
        GradingScale(scale_id=scale_id, scale_name="ICDR_0_4", disease_type="DR")
    )
    await _make_chain(chain_id)

    await upsert_disease_grading(
        DiseaseGrading(
            grading_id=grading_id,
            image_id=image_id,
            disease_type="DR",
            scale_id=scale_id,
            original_grade="2",
            scaled_grade=None,  # trigger fills this in
            provenance_chain_id=chain_id,
        )
    )

    # Trigger set scaled_grade; no listener → no audit row yet.
    assert len(await get_transformations_for_chain(chain_id)) == 0

    await reconcile_grade_conversions()
    after_first = await get_transformations_for_chain(chain_id)
    assert len(after_first) == 1
    assert after_first[0].operation_type == TRANSFORMATION_TYPE

    # Idempotent: a second sweep does not duplicate.
    await reconcile_grade_conversions()
    assert len(await get_transformations_for_chain(chain_id)) == 1

    # And the lineage for the image surfaces the chain + its transformation.
    lineage = await get_lineage_for_image(image_id)
    chains_seen = {entry["chain"].chain_id for entry in lineage}
    assert chain_id in chains_seen
    entry = next(e for e in lineage if e["chain"].chain_id == chain_id)
    assert any(t.operation_type == TRANSFORMATION_TYPE for t in entry["transformations"])
