"""
CaptionEngine: Synthesize text captions from annotations and dictionary definitions.

Generates human-readable captions from disease gradings, keywords,
classification labels, localization structures, and segmentation structures
using the expert knowledge definitions dictionary.
"""

import json
import random
import re
from typing import Any, Optional

from chaksudb.common.dictionary import (
    _CLASS_NAME_ALIASES,
    _GRADE_LABEL_ALIASES,
    _NORMAL_PREFIXES,
    _TEMPLATES,
)

# Pre-compiled regex patterns used in hot paths (called per-row)
_RE_GRADER_SUFFIX = re.compile(r"_g\d+$")
_RE_INSTANCE_SUFFIX = re.compile(r"_\d+$")

# Segmentation/localization annotation types that are data artifacts, not clinical findings
_NON_CLINICAL_STRUCTURES = frozenset({"attention_map", "fundus_area"})

# Classification class_names to skip (not useful for captions)
_SKIP_CLASS_NAMES = frozenset({"image_quality"})

_NORMAL_MARKER = "normal fundus"


def _snake_to_readable(name: str) -> str:
    """Convert snake_case to readable form: 'cotton_wool_spots' -> 'cotton wool spots'."""
    return name.replace("_", " ")


def _is_normal_finding(finding: str) -> bool:
    """Return True when a finding string represents a healthy / disease-absent state."""
    f = finding.lower().strip()
    return any(f.startswith(p) for p in _NORMAL_PREFIXES)


def _dedup(items: list[str]) -> list[str]:
    """Deduplicate a list while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class CaptionEngine:
    """Generate text captions from annotation data and dictionary definitions."""

    def __init__(
        self,
        definitions: dict[str, list[str]],
        abbreviations: dict[str, str],
    ):
        self.definitions = definitions
        self.abbreviations = abbreviations
        self._norm_defs: dict[str, list[str]] = {
            k.lower().strip(): v for k, v in definitions.items()
        }

    def _lookup(self, key: str) -> list[str] | None:
        """Look up definitions by key, trying aliases, normalisation, and plural/singular."""
        k = key.lower().strip()
        if k in self._norm_defs:
            return self._norm_defs[k]
        alias = _GRADE_LABEL_ALIASES.get(k) or _CLASS_NAME_ALIASES.get(k)
        if alias and alias.lower() in self._norm_defs:
            return self._norm_defs[alias.lower()]
        spaced = _snake_to_readable(k)
        if spaced in self._norm_defs:
            return self._norm_defs[spaced]
        if k.endswith("s"):
            singular = k[:-1]
            if singular in self._norm_defs:
                return self._norm_defs[singular]
        else:
            plural = k + "s"
            if plural in self._norm_defs:
                return self._norm_defs[plural]
        return None

    def _enrich(self, finding: str, fallback_key: str | None = None) -> str:
        """Enrich a finding string with its definition, returning a formatted string.

        Tries to look up the finding itself first, then the fallback_key.
        Avoids tautological enrichment (where description == finding).
        """
        descs = self._lookup(finding)
        if not descs and fallback_key:
            descs = self._lookup(fallback_key)
        if descs:
            detail = descs[0]
            if detail.lower() != finding.lower():
                return f"{finding} ({detail})"
        return finding

    def _normal_description(self) -> str:
        """Return the standard 'normal fundus' description string."""
        normal_descs = self._lookup("normal")
        if normal_descs:
            return f"{_NORMAL_MARKER}: " + ", ".join(normal_descs)
        return _NORMAL_MARKER

    # ------------------------------------------------------------------
    # Per-source methods
    # ------------------------------------------------------------------

    def from_grading(
        self,
        disease_type: str,
        grade: Optional[int],
        original_grade: Optional[str] = None,
    ) -> Optional[str]:
        """Generate caption from a disease grading annotation."""
        if grade is None:
            return None

        lookup_keys = []
        if original_grade:
            lookup_keys.append(original_grade.strip())
        lookup_keys.append(f"{disease_type} grade {grade}")

        descriptions: list[str] | None = None
        for key in lookup_keys:
            descriptions = self._lookup(key)
            if descriptions:
                break

        if descriptions:
            detail = ", ".join(descriptions)
            return f"{original_grade or disease_type}: {detail}"

        label = original_grade or f"{disease_type} grade {grade}"
        return label

    def from_grading_data(self, grade_data: Any) -> Optional[str]:
        """Generate caption from the aggregated grading JSONB array."""
        if isinstance(grade_data, str):
            try:
                grade_data = json.loads(grade_data)
            except (json.JSONDecodeError, ValueError):
                return None
        if not grade_data or not isinstance(grade_data, list):
            return None

        parts: list[str] = []
        has_normal = False
        for entry in grade_data:
            if not isinstance(entry, dict):
                continue
            label = (entry.get("grade_label") or entry.get("original_grade") or "").strip()
            disease = (entry.get("disease_type") or "").strip()
            if not label and not disease:
                continue

            resolved = label
            alias = _GRADE_LABEL_ALIASES.get(label.lower().strip()) if label else None
            if alias:
                resolved = alias

            if _is_normal_finding(resolved):
                has_normal = True
                continue

            descs = self._lookup(label) if label else None
            if descs:
                detail = ", ".join(descs)
                parts.append(f"{resolved} ({detail})")
            elif resolved != label and resolved != label.lower():
                parts.append(resolved)
            elif label:
                parts.append(label)
            elif disease:
                parts.append(disease)

        if has_normal and not parts:
            parts.append(self._normal_description())

        if not parts:
            return None
        return "; ".join(_dedup(parts))

    def from_keywords(self, keywords: list[str]) -> Optional[str]:
        """Generate caption from keyword annotations using definitions."""
        if not keywords:
            return None

        parts: list[str] = []
        for kw in keywords:
            descs = self._lookup(kw)
            if descs:
                parts.append(f"{kw} ({descs[0]})")
            else:
                parts.append(kw)

        return ", ".join(parts)

    def from_classification(self, class_labels: dict[str, str]) -> Optional[str]:
        """Generate caption from classification label dict {class_name: class_label}."""
        if not class_labels:
            return None

        parts: list[str] = []
        for class_name, class_label in class_labels.items():
            descs = self._lookup(class_name)
            if descs:
                parts.append(f"{_snake_to_readable(class_name)} ({descs[0]})")
            else:
                parts.append(f"{_snake_to_readable(class_name)}: {class_label}")

        return ", ".join(parts)

    def _extract_class_finding(self, class_name: str, class_value: Any) -> Optional[str]:
        """Extract a human-readable finding from a classification entry.

        Handles the nested JSONB structures stored in classification_annotations:
        - {"glaucoma": true/false} -> "glaucoma" / "no glaucoma"
        - {"cdr": 0.7, "glaucoma": false} -> picks the boolean/label field
        - {"class_label": "red_lesions"} -> "red lesions"
        - scalar values (True/False/int) -> direct interpretation
        """
        readable_name = _snake_to_readable(class_name)

        # Scalar boolean
        if isinstance(class_value, bool):
            return readable_name if class_value else f"no {readable_name}"

        # Scalar int/float (e.g. a grade)
        if isinstance(class_value, (int, float)):
            return f"{readable_name} (grade {class_value})"

        # Scalar string
        if isinstance(class_value, str):
            return _snake_to_readable(class_value)

        # Dict — the most common case from the DB
        if isinstance(class_value, dict):
            # Unwrap single-key wrapper dicts: {"lesion_type": {"class_label": "x"}}
            if len(class_value) == 1:
                inner = next(iter(class_value.values()))
                if isinstance(inner, (dict, bool, str, int, float)):
                    return self._extract_class_finding(class_name, inner)

            # Look for a boolean field or class_label
            bool_key = None
            label_key = None
            for k, v in class_value.items():
                if isinstance(v, bool):
                    bool_key = k
                if k == "class_label":
                    label_key = k

            if label_key is not None:
                raw_label = class_value[label_key]
                if isinstance(raw_label, dict) and "class_label" in raw_label:
                    raw_label = raw_label["class_label"]
                return _snake_to_readable(str(raw_label))

            if bool_key is not None:
                return readable_name if class_value[bool_key] else f"no {readable_name}"

            # Fallback: pick first meaningful value
            for v in class_value.values():
                if isinstance(v, str):
                    return _snake_to_readable(v)
                if isinstance(v, bool):
                    return readable_name if v else f"no {readable_name}"

        return readable_name

    @staticmethod
    def _is_per_expert_multilabel(class_name: str, class_value: Any) -> bool:
        """Detect per-grader multi-label feature entries like 'glaucoma_features_g1'."""
        if not isinstance(class_value, dict):
            return False
        if not _RE_GRADER_SUFFIX.search(class_name):
            return False
        return all(isinstance(v, bool) for v in class_value.values())

    @staticmethod
    def _merge_expert_features(entries: dict[str, dict[str, bool]]) -> set[str]:
        """Merge multi-label feature dicts from multiple graders.

        Returns the set of feature names that are positive in *any* grader.
        """
        positive: set[str] = set()
        for features in entries.values():
            for feat, val in features.items():
                if val:
                    positive.add(feat)
        return positive

    def from_classification_data(self, class_data: Any) -> Optional[str]:
        """Generate caption from the aggregated classification JSONB object."""
        if isinstance(class_data, str):
            try:
                class_data = json.loads(class_data)
            except (json.JSONDecodeError, ValueError):
                return None
        if not class_data or not isinstance(class_data, dict):
            return None

        expert_groups: dict[str, dict[str, dict[str, bool]]] = {}
        regular_entries: dict[str, Any] = {}

        for class_name, class_value in class_data.items():
            if class_value is None or class_name in _SKIP_CLASS_NAMES:
                continue
            if self._is_per_expert_multilabel(class_name, class_value):
                base = _RE_GRADER_SUFFIX.sub("", class_name)
                expert_groups.setdefault(base, {})[class_name] = class_value
            else:
                regular_entries[class_name] = class_value

        parts: list[str] = []
        has_normal = False

        for class_name, class_value in regular_entries.items():
            finding = self._extract_class_finding(class_name, class_value)
            if finding is None:
                continue
            if _is_normal_finding(finding):
                has_normal = True
                continue
            parts.append(self._enrich(finding, fallback_key=class_name))

        for grader_entries in expert_groups.values():
            positive_features = self._merge_expert_features(grader_entries)
            if positive_features:
                readable = [_snake_to_readable(f) for f in sorted(positive_features)]
                parts.append("with features: " + ", ".join(readable))

        if has_normal and not parts:
            parts.append(self._normal_description())

        if not parts:
            return None
        return ", ".join(parts)

    def from_structures(self, structures: Any) -> Optional[str]:
        """Generate caption from a list or comma-separated string of structure names.

        Filters out non-clinical items (attention maps, area masks) and collapses
        numbered instance IDs (``ma_1, ma_2, ...``) into a single base type
        (``ma``).
        """
        if not structures:
            return None

        if isinstance(structures, str):
            items = [s.strip() for s in structures.split(",") if s.strip()]
        elif isinstance(structures, list):
            items = [str(s).strip() for s in structures if s]
        else:
            return None

        if not items:
            return None

        # Collapse numbered instances and filter non-clinical entries
        base_types: dict[str, None] = {}  # ordered set
        for s in items:
            if s.lower() in _NON_CLINICAL_STRUCTURES:
                continue
            base = _RE_INSTANCE_SUFFIX.sub("", s)
            if base not in base_types:
                base_types[base] = None

        if not base_types:
            return None

        parts: list[str] = []
        for s in base_types:
            descs = self._lookup(s)
            if descs:
                parts.append(f"{_snake_to_readable(s)} ({descs[0]})")
            else:
                parts.append(_snake_to_readable(s))

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Synthesis: combine all sources into one caption
    # ------------------------------------------------------------------

    def synthesize(self, row: dict[str, Any], mode: str = "all") -> Optional[str]:
        """Synthesize a rich caption from all available annotation columns in a row.

        Produces a single natural-language sentence combining all annotation
        sources, enriched by the definitions dictionary.
        """
        findings: list[str] = []

        if mode in ("grading", "synthetic", "all"):
            cap = self.from_grading_data(row.get("caption_grade_data"))
            if cap:
                findings.append(cap)

        if mode in ("classification", "synthetic", "all"):
            cap = self.from_classification_data(row.get("caption_class_data"))
            if cap:
                findings.append(cap)

        if mode in ("synthetic", "all"):
            cap = self.from_structures(row.get("caption_loc_structures"))
            if cap:
                findings.append(cap)

        if mode in ("synthetic", "all"):
            cap = self.from_structures(row.get("caption_seg_structures"))
            if cap:
                findings.append(cap)

        if mode in ("keyword", "all"):
            keywords = row.get("caption_keywords")
            if keywords:
                if isinstance(keywords, str):
                    try:
                        keywords = json.loads(keywords)
                    except (json.JSONDecodeError, ValueError):
                        keywords = None
                if keywords and isinstance(keywords, list):
                    cap = self.from_keywords(keywords)
                    if cap:
                        findings.append(cap)

        if mode in ("clinical", "all"):
            clinical = row.get("caption_clinical_text")
            if clinical:
                findings.append(str(clinical))

        if not findings:
            return None

        # If any source contributes a pathological finding, drop "normal fundus"
        # entries from other sources to avoid contradictions
        has_pathology = any(_NORMAL_MARKER not in f for f in findings)
        if has_pathology:
            findings = [f for f in findings if _NORMAL_MARKER not in f]

        if not findings:
            return None

        combined = "; ".join(_dedup(findings))
        return f"A fundus photograph showing {combined}."

    # ------------------------------------------------------------------
    # Variant generation
    # ------------------------------------------------------------------

    def generate_variants(
        self,
        base_caption: str,
        n_variants: int = 3,
        *,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Generate multiple caption variants for training augmentation.

        Rewrites the base caption using different template prefixes.
        """
        if not base_caption:
            return []

        findings = base_caption
        for marker in ["showing ", "with findings of ", "with ", "include ", "revealing ", "demonstrates "]:
            idx = base_caption.lower().find(marker)
            if idx >= 0:
                findings = base_caption[idx + len(marker):].rstrip(".")
                break

        rng = random.Random(seed)
        templates = list(_TEMPLATES)
        rng.shuffle(templates)

        variants: list[str] = []
        for tmpl in templates[:n_variants]:
            variants.append(tmpl.format(findings=findings))

        return variants
