"""
Consensus annotation database operations.
"""

import psycopg

from chaksudb.db.connection import get_connection
from chaksudb.db.models import ConsensusAnnotation
from chaksudb.db.queries.helpers import _prepare_upsert_query, _serialize_jsonb


async def upsert_consensus_annotation(consensus: ConsensusAnnotation) -> None:
    """Upsert a consensus annotation record."""
    async with get_connection() as conn:
        columns = [
            "consensus_id",
            "image_id",
            "annotation_task",
            "consensus_method",
            "expert_annotation_ids",
            "consensus_value",
            "agreement_score",
            "disagreement_details",
            "adjudicator_id",
            "created_at",
        ]
        values = (
            consensus.consensus_id,
            consensus.image_id,
            consensus.annotation_task,
            consensus.consensus_method,
            [str(eid) for eid in (consensus.expert_annotation_ids or [])],
            _serialize_jsonb(consensus.consensus_value),
            consensus.agreement_score,
            _serialize_jsonb(consensus.disagreement_details),
            consensus.adjudicator_id,
            consensus.created_at,
        )
        query = _prepare_upsert_query(
            "consensus_annotations",
            columns,
            conflict_target=["consensus_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)
