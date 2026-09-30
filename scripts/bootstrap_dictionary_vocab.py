#!/usr/bin/env python3
"""
Bootstrap script: seed keyword_vocabulary with all terms from chaksudb/common/dictionary.py.

Creates a synthetic EXPERT_DICTIONARY dataset (no images) and inserts every key from
`definitions` as a diagnostic_keywords vocabulary entry with category='expert_definition'.
Also inserts all synonym/description phrases as separate entries with category='expert_synonym'.

Idempotent — safe to re-run at any time.

Usage:
    uv run scripts/bootstrap_dictionary_vocab.py
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chaksudb.common.dictionary import definitions
from chaksudb.db import close_pool
from chaksudb.db.models import Dataset, KeywordVocabulary
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.queries.annotation_types import upsert_keyword_vocabulary
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid, generate_keyword_uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DICT_DATASET_NAME = "EXPERT_DICTIONARY"


async def bootstrap_dictionary_vocab() -> dict[str, int]:
    dataset_id = generate_dataset_uuid(DICT_DATASET_NAME)

    dataset = Dataset(
        dataset_id=dataset_id,
        dataset_name=DICT_DATASET_NAME,
        source_url=None,
        license=None,
        modality_types=[],
    )
    await upsert_dataset(dataset)
    logger.info("Upserted dataset %s (%s)", DICT_DATASET_NAME, dataset_id)

    inserted_keys = 0
    inserted_synonyms = 0

    for condition, synonyms in definitions.items():
        term = condition.lower().strip()
        kv = KeywordVocabulary(
            keyword_id=generate_keyword_uuid(dataset_id, term, "diagnostic_keywords"),
            keyword_term=term,
            keyword_source="diagnostic_keywords",
            category="expert_definition",
            dataset_id=dataset_id,
        )
        await upsert_keyword_vocabulary(kv)
        inserted_keys += 1

        for phrase in synonyms:
            phrase_lower = phrase.lower().strip()
            if phrase_lower == term:
                continue
            sv = KeywordVocabulary(
                keyword_id=generate_keyword_uuid(dataset_id, phrase_lower, "diagnostic_keywords"),
                keyword_term=phrase_lower,
                keyword_source="diagnostic_keywords",
                category="expert_synonym",
                dataset_id=dataset_id,
            )
            await upsert_keyword_vocabulary(sv)
            inserted_synonyms += 1

    return {"condition_terms": inserted_keys, "synonym_terms": inserted_synonyms}


async def main() -> int:
    try:
        stats = await bootstrap_dictionary_vocab()
        print("\nBootstrap completed.")
        print(f"  Condition terms: {stats['condition_terms']}")
        print(f"  Synonym terms:   {stats['synonym_terms']}")
        print(f"  Total:           {stats['condition_terms'] + stats['synonym_terms']}")
        return 0
    except Exception:
        logger.exception("Bootstrap failed")
        return 1
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
