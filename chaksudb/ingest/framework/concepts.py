"""
Canonical clinical concept vocabulary.

A *concept* is a dataset-independent clinical entity (e.g. ``glaucoma``) that may be
expressed through different classification shapes across datasets — binary in AIROGS,
a sub-key of a multi-label vector in PAPILA, the winning class of a multi-class task
elsewhere. This module is the single source of truth that both:

  - ingest uses to normalize a dataset's free-text label into a canonical concept, and
  - export uses to resolve a cross-cutting "give me <concept>-positive images" filter,

so the two never drift. Add a dataset's aliases here once and both sides understand it.
"""

from __future__ import annotations

# canonical concept -> set of lowercase aliases that mean it
CONCEPT_ALIASES: dict[str, set[str]] = {
    "glaucoma": {
        "glaucoma",
        "rg",
        "referable_glaucoma",
        "glaucoma_diagnosis",
        "glaucoma_classification",
        "suspect_glaucoma",
    },
    "DR": {
        "dr",
        "diabetic_retinopathy",
        "diabetes",
        "diabetic",
    },
    "AMD": {
        "amd",
        "armd",
        "age_related_macular_degeneration",
    },
    "DME": {
        "dme",
        "diabetic_macular_edema",
        "macular_edema",
    },
    "cataract": {
        "cataract",
    },
    "normal": {
        "normal",
        "healthy",
        "no_disease",
    },
}

# reverse index: alias -> canonical concept
_ALIAS_TO_CONCEPT: dict[str, str] = {
    alias: concept
    for concept, aliases in CONCEPT_ALIASES.items()
    for alias in aliases
}

# also let the canonical name itself resolve case-insensitively
for _concept in CONCEPT_ALIASES:
    _ALIAS_TO_CONCEPT.setdefault(_concept.lower(), _concept)


def to_concept(name: str | None) -> str | None:
    """Resolve a free-text class/label name to a canonical concept, or None if unknown.

    Unknown names intentionally return None rather than guessing — a row with no concept
    simply does not participate in cross-concept filtering.
    """
    if name is None:
        return None
    return _ALIAS_TO_CONCEPT.get(name.strip().lower())


def normalize_class_name(name: str) -> str:
    """Return the canonical concept name if known, else the stripped input unchanged.

    Used at ingest to keep class_name values consistent (e.g. ``armd`` -> ``AMD``).
    """
    return to_concept(name) or name.strip()


def all_concepts() -> list[str]:
    """All canonical concept names (for discovery/validation)."""
    return sorted(CONCEPT_ALIASES)
