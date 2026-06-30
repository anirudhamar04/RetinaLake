"""
Bootstrap grading scale mappings from SUSTech-SYSU dataset.

This script analyzes the SUSTech-SYSU dataset which contains simultaneous
gradings in three different scales (ICDR, AAO, Scottish) to learn the
mappings between them.

The script can be run on both the sampled subset and the full dataset,
deriving mappings from whatever data is available.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from chaksudb.config.config import constants, get_data_root
from chaksudb.db.models import GradingScale, GradingScaleMapping
from chaksudb.db.queries.grading import (
    upsert_grading_scale,
    upsert_grading_scale_mapping,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_grading_scale_uuid,
    generate_grading_scale_mapping_uuid,
)

logger = logging.getLogger(__name__)


# ============================================
# Scale Definitions
# ============================================

SCALE_DEFINITIONS = {
    "ICDR_0_4": {
        "scale_name": "ICDR_0_4",
        "disease_type": "DR",
        "scale_description": "International Clinical DR Severity Scale (0-4)",
        "min_value": 0,
        "max_value": 4,
        "value_labels": {
            "0": "No DR",
            "1": "Mild NPDR",
            "2": "Moderate NPDR",
            "3": "Severe NPDR",
            "4": "PDR (Proliferative DR)",
        },
    },
    "ICDR_0_5": {
        "scale_name": "ICDR_0_5",
        "disease_type": "DR",
        "scale_description": "International Clinical DR Severity Scale (0-5, includes ungradable)",
        "min_value": 0,
        "max_value": 5,
        "value_labels": {
            "0": "No DR",
            "1": "Mild NPDR",
            "2": "Moderate NPDR",
            "3": "Severe NPDR",
            "4": "PDR (Proliferative DR)",
            "5": "Ungradable (poor quality, laser scars)",
        },
    },
    "AAO": {
        "scale_name": "AAO",
        "disease_type": "DR",
        "scale_description": "American Academy of Ophthalmology DR grading scale",
        "min_value": 0,
        "max_value": 5,
        "value_labels": {
            "0": "No DR",
            "1": "Mild NPDR",
            "2": "Moderate NPDR",
            "3": "Severe NPDR",
            "4": "PDR",
            "5": "Advanced PDR or ungradable",
        },
    },
    "Scottish": {
        "scale_name": "Scottish",
        "disease_type": "DR",
        "scale_description": "Scottish DR grading protocol",
        "min_value": 0,
        "max_value": 5,
        "value_labels": {
            "0": "No DR",
            "1": "Mild background DR",
            "2": "Observable background DR",
            "3": "Referable DR",
            "4": "Proliferative DR",
            "5": "Ungradable",
        },
    },
    "PARAGUAY_DR_7_level": {
        "scale_name": "PARAGUAY_DR_7_level",
        "disease_type": "DR",
        "scale_description": "PARAGUAY 7-level DR grading (0-4, with detailed NPDR stages)",
        "min_value": 0,
        "max_value": 7,
        "value_labels": {
            "1. No DR signs": "No DR signs",
            "2. Mild (or early) NPDR": "Mild (or early) NPDR",
            "3. Moderate NPDR": "Moderate NPDR",
            "4. Severe NPDR": "Severe NPDR",
            "5. Very Severe NPDR": "Very Severe NPDR",
            "6. PDR": "PDR",
            "7. Advanced PDR": "Advanced PDR",
        },
    },
    "1000x39_DR_0_3": {
        "scale_name": "1000x39_DR_0_3",
        "disease_type": "DR",
        "scale_description": "1000x39 custom 4-level DR grading (0-3)",
        "min_value": 0,
        "max_value": 3,
        "value_labels": {
            "0": "No DR (Normal)",
            "1": "DR1",
            "2": "DR2",
            "3": "DR3",
        },
    },
    # Canonical DME scale that per-dataset DME scales map into.
    "DME_0_2": {
        "scale_name": "DME_0_2",
        "disease_type": "DME",
        "scale_description": "Canonical Diabetic Macular Edema severity scale (0-2)",
        "min_value": 0,
        "max_value": 2,
        "value_labels": {
            "0": "No DME",
            "1": "Mild/Moderate DME",
            "2": "Severe DME",
        },
    },
    "IDRID_DME_0_2": {
        "scale_name": "IDRID_DME_0_2",
        "disease_type": "DME",
        "scale_description": "IDRiD DME risk grading (0-2)",
        "min_value": 0,
        "max_value": 2,
        "value_labels": {
            "0": "No DME",
            "1": "Mild/Moderate DME",
            "2": "Severe DME",
        },
    },
    "MESSIDOR2_DME_0_2": {
        "scale_name": "MESSIDOR2_DME_0_2",
        "disease_type": "DME",
        "scale_description": "MESSIDOR-2 DME risk grading (0-2)",
        "min_value": 0,
        "max_value": 2,
        "value_labels": {
            "0": "no_risk",
            "1": "low_risk",
            "2": "high_risk",
        },
    },
}


# ============================================
# Analysis Functions
# ============================================


def analyze_scale_mappings(
    csv_path: Path,
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Analyze SUSTech-SYSU drLabels.csv to extract scale mappings.

    Args:
        csv_path: Path to drLabels.csv file

    Returns:
        Dictionary mapping source_scale -> target_scale -> source_value -> target_value
        Example: {"ICDR": {"AAO": {"0": 0, "1": 1, ...}}}
    """
    logger.info(f"Analyzing scale mappings from {csv_path}")

    # Track all observed mappings: source_scale -> source_value -> [target_values]
    mapping_observations: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Column names from SUSTech-SYSU dataset
            icdr_grade = row.get(
                "DR_grade(International_Clinical_DR_Severity_Scale)",
                row.get("DR_grade(ICDR)"),  # Fallback for shorter name
            )
            aao_grade = row.get(
                "DR_grade(American_Academy_of_Ophthalmology)",
                row.get("DR_grade(AAO)"),  # Fallback
            )
            scottish_grade = row.get(
                "DR_grade(Scottish_DR_grading_protocol)",
                row.get("DR_grade(Scottish)"),  # Fallback
            )

            if not icdr_grade or not aao_grade or not scottish_grade:
                logger.warning(f"Missing grade data in row: {row}")
                continue

            # Convert to strings with error handling
            try:
                icdr = str(int(float(icdr_grade)))
                aao = str(int(float(aao_grade)))
                scottish = str(int(float(scottish_grade)))
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid grade values - ICDR: {icdr_grade}, "
                    f"AAO: {aao_grade}, Scottish: {scottish_grade}. Error: {e}"
                )
                continue

            # Record mappings in both directions
            # ICDR -> AAO
            mapping_observations["ICDR_0_4"]["AAO"][icdr].append(int(aao))
            # ICDR -> Scottish
            mapping_observations["ICDR_0_4"]["Scottish"][icdr].append(int(scottish))
            # AAO -> ICDR
            mapping_observations["AAO"]["ICDR_0_4"][aao].append(int(icdr))
            # AAO -> Scottish
            mapping_observations["AAO"]["Scottish"][aao].append(int(scottish))
            # Scottish -> ICDR
            mapping_observations["Scottish"]["ICDR_0_4"][scottish].append(int(icdr))
            # Scottish -> AAO
            mapping_observations["Scottish"]["AAO"][scottish].append(int(aao))

    logger.info(f"Observed mapping data from {csv_path}")
    return mapping_observations


def validate_mappings(
    mapping_observations: dict[str, dict[str, dict[str, list[int]]]],
) -> dict[str, dict[str, dict[str, tuple[int, str]]]]:
    """
    Validate and consolidate mapping observations.

    Args:
        mapping_observations: Raw observations from analyze_scale_mappings

    Returns:
        Validated mappings: source_scale -> target_scale -> source_value -> (target_value, confidence)
        Confidence is one of: 'exact', 'approximate', 'manual_review_required'
    """
    logger.info("Validating scale mappings")

    validated_mappings: dict[str, dict[str, dict[str, tuple[int, str]]]] = {}

    for source_scale, targets in mapping_observations.items():
        validated_mappings[source_scale] = {}

        for target_scale, source_values in targets.items():
            validated_mappings[source_scale][target_scale] = {}

            for source_value, target_values in source_values.items():
                # Count occurrences of each target value
                value_counts: dict[int, int] = defaultdict(int)
                for val in target_values:
                    value_counts[val] += 1

                total_count = len(target_values)
                most_common_value = max(value_counts.items(), key=lambda x: x[1])
                most_common_count = most_common_value[1]
                consensus_value = most_common_value[0]

                # Determine confidence
                if most_common_count == total_count:
                    # Perfect consistency
                    confidence = "exact"
                elif most_common_count / total_count >= 0.85:
                    # Strong consensus (85%+)
                    confidence = "exact"
                    logger.info(
                        f"Strong consensus for {source_scale}:{source_value} -> "
                        f"{target_scale}:{consensus_value} "
                        f"({most_common_count}/{total_count} = {most_common_count/total_count:.1%})"
                    )
                elif most_common_count / total_count >= 0.65:
                    # Reasonable consensus (65-85%)
                    confidence = "approximate"
                    logger.warning(
                        f"Approximate mapping for {source_scale}:{source_value} -> "
                        f"{target_scale}:{consensus_value} "
                        f"({most_common_count}/{total_count} = {most_common_count/total_count:.1%})"
                    )
                else:
                    # Low consensus (<65%)  
                    confidence = "manual_review_required"
                    logger.error(
                        f"Inconsistent mapping for {source_scale}:{source_value} -> "
                        f"{target_scale}: {dict(value_counts)} "
                        f"({most_common_count}/{total_count} = {most_common_count/total_count:.1%})"
                    )

                validated_mappings[source_scale][target_scale][source_value] = (
                    consensus_value,
                    confidence,
                )

    return validated_mappings


async def bootstrap_grading_scales(
    sustech_data_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> dict[str, str]:
    """
    Bootstrap grading scales and mappings from SUSTech-SYSU dataset.

    This function:
    1. Registers all known grading scales (ICDR_0_4, ICDR_0_5, AAO, Scottish, PARAGUAY_DR_7_level)
    2. Analyzes drLabels.csv to learn mappings between scales
    3. Handles c5_DR_reclassified.csv for grade 5 cases (if exists)
    4. Adds manual PARAGUAY -> ICDR_0_4 mappings
    5. Stores validated mappings in the database

    Args:
        sustech_data_dir: Path to SUSTech-SYSU dataset directory.
                          If None, uses default from config.
        force_reload: If True, re-analyzes even if mappings exist

    Returns:
        Dictionary with summary statistics
    """
    logger.info("Starting grading scale bootstrap")

    # Determine data directory
    if sustech_data_dir is None:
        data_root = get_data_root()
        sustech_data_dir = data_root / "30_SUSTech-SYSU"

    # Check if SUSTech-SYSU data is available
    sustech_available = False
    if sustech_data_dir.exists():
        dr_labels_path = sustech_data_dir / "drLabels.csv"
        if dr_labels_path.exists():
            sustech_available = True
            logger.info(f"SUSTech-SYSU data found at {sustech_data_dir}")
        else:
            logger.warning(f"SUSTech-SYSU directory exists but drLabels.csv not found")
    else:
        logger.warning(f"SUSTech-SYSU directory not found: {sustech_data_dir}")
    
    if not sustech_available:
        logger.info("Will register scales and manual mappings only (no SUSTech-SYSU analysis)")

    # ============================================
    # Step 1: Register all grading scales
    # ============================================
    logger.info("Registering grading scales")

    registered_scales = {}
    for scale_key, scale_def in SCALE_DEFINITIONS.items():
        scale_id = generate_grading_scale_uuid(
            scale_def["scale_name"],
            scale_def["disease_type"],
        )

        grading_scale = GradingScale(
            scale_id=scale_id,
            scale_name=scale_def["scale_name"],
            disease_type=scale_def["disease_type"],
            scale_description=scale_def["scale_description"],
            min_value=scale_def["min_value"],
            max_value=scale_def["max_value"],
            value_labels=scale_def["value_labels"],
        )

        await upsert_grading_scale(grading_scale)
        registered_scales[scale_key] = scale_id
        logger.info(f"Registered scale: {scale_def['scale_name']}")

    # ============================================
    # Step 2: Analyze drLabels.csv for mappings (if available)
    # ============================================
    validated_mappings = {}
    
    if sustech_available:
        mapping_observations = analyze_scale_mappings(dr_labels_path)

        # ============================================
        # Step 3: Validate mappings
        # ============================================
        validated_mappings = validate_mappings(mapping_observations)

        # ============================================
        # Step 4: Handle c5_DR_reclassified.csv (grade 5 cases)
        # ============================================
        c5_path = sustech_data_dir / "c5_DR_reclassified.csv"
        if c5_path.exists():
            logger.info(f"Processing grade 5 reclassifications from {c5_path}")
            c5_observations = analyze_scale_mappings(c5_path)
            c5_validated = validate_mappings(c5_observations)

            # Merge with main mappings
            for source_scale, targets in c5_validated.items():
                if source_scale not in validated_mappings:
                    validated_mappings[source_scale] = {}
                for target_scale, mappings in targets.items():
                    if target_scale not in validated_mappings[source_scale]:
                        validated_mappings[source_scale][target_scale] = {}
                    # Only add grade 5 mappings (don't override existing)
                    for source_val, (target_val, confidence) in mappings.items():
                        if int(source_val) >= 4:  # Grade 4 or 5
                            validated_mappings[source_scale][target_scale][source_val] = (
                                target_val,
                                confidence,
                            )
    else:
        logger.info("Skipping SUSTech-SYSU analysis (data not available)")

    # ============================================
    # Step 5: Add manual ICDR_0_5 -> ICDR_0_4 mappings
    # ============================================
    logger.info("Adding manual ICDR_0_5 -> ICDR_0_4 mappings")

    if "ICDR_0_5" not in validated_mappings:
        validated_mappings["ICDR_0_5"] = {}
    if "ICDR_0_4" not in validated_mappings["ICDR_0_5"]:
        validated_mappings["ICDR_0_5"]["ICDR_0_4"] = {}

    # Grades 0-4 are identical across both scales; grade 5 (Ungradable) has no ICDR_0_4 equivalent
    icdr05_to_icdr04_mappings = {
        "0": (0, "exact"),
        "1": (1, "exact"),
        "2": (2, "exact"),
        "3": (3, "exact"),
        "4": (4, "exact"),
    }

    for source_val, (target_val, confidence) in icdr05_to_icdr04_mappings.items():
        validated_mappings["ICDR_0_5"]["ICDR_0_4"][source_val] = (target_val, confidence)
        logger.info(f"  ICDR_0_5 '{source_val}' -> ICDR_0_4 {target_val} (confidence: {confidence})")

    # ============================================
    # Step 6: Add manual PARAGUAY -> ICDR_0_4 mappings
    # ============================================
    logger.info("Adding manual PARAGUAY -> ICDR_0_4 mappings")

    # Initialize PARAGUAY mappings structure
    if "PARAGUAY_DR_7_level" not in validated_mappings:
        validated_mappings["PARAGUAY_DR_7_level"] = {}
    if "ICDR_0_4" not in validated_mappings["PARAGUAY_DR_7_level"]:
        validated_mappings["PARAGUAY_DR_7_level"]["ICDR_0_4"] = {}

    # Manual mappings based on clinical correspondence
    paraguay_to_icdr_mappings = {
        "1. No DR signs": (0, "exact"),              # No DR → No DR
        "2. Mild (or early) NPDR": (1, "exact"),     # Mild NPDR → Mild NPDR
        "3. Moderate NPDR": (2, "exact"),            # Moderate NPDR → Moderate NPDR
        "4. Severe NPDR": (3, "exact"),              # Severe NPDR → Severe NPDR
        "5. Very Severe NPDR": (3, "approximate"),   # Very Severe NPDR → Severe NPDR (approximate)
        "6. PDR": (4, "exact"),                      # PDR → PDR
        "7. Advanced PDR": (4, "exact"),             # Advanced PDR → PDR
    }

    for source_val, (target_val, confidence) in paraguay_to_icdr_mappings.items():
        validated_mappings["PARAGUAY_DR_7_level"]["ICDR_0_4"][source_val] = (target_val, confidence)
        logger.info(f"  PARAGUAY '{source_val}' -> ICDR {target_val} (confidence: {confidence})")

    # ============================================
    # Step 6b: Map 1000x39 custom DR scale -> ICDR_0_4
    # ============================================
    logger.info("Adding manual 1000x39_DR_0_3 -> ICDR_0_4 mappings")
    validated_mappings.setdefault("1000x39_DR_0_3", {})["ICDR_0_4"] = {
        "0": (0, "exact"),
        "1": (1, "approximate"),  # DR1 ~ Mild NPDR
        "2": (2, "approximate"),  # DR2 ~ Moderate NPDR
        "3": (4, "approximate"),  # DR3 ~ PDR (collapses severe/proliferative)
    }

    # ============================================
    # Step 6c: Map per-dataset DME scales -> canonical DME_0_2
    # ============================================
    logger.info("Adding DME scale mappings -> DME_0_2")
    dme_identity = {"0": (0, "exact"), "1": (1, "exact"), "2": (2, "exact")}
    validated_mappings.setdefault("IDRID_DME_0_2", {})["DME_0_2"] = dict(dme_identity)
    validated_mappings.setdefault("MESSIDOR2_DME_0_2", {})["DME_0_2"] = dict(dme_identity)

    # ============================================
    # Step 7: Store mappings in database
    # ============================================
    logger.info("Storing scale mappings in database")

    mapping_stats = {
        "total_mappings": 0,
        "exact_mappings": 0,
        "approximate_mappings": 0,
        "manual_review_required": 0,
    }

    for source_scale, targets in validated_mappings.items():
        source_scale_id = registered_scales[source_scale]

        for target_scale, mappings in targets.items():
            target_scale_id = registered_scales[target_scale]

            for source_value, (target_value, confidence) in mappings.items():
                mapping_id = generate_grading_scale_mapping_uuid(
                    source_scale_id,
                    target_scale_id,
                    source_value,
                )

                mapping = GradingScaleMapping(
                    mapping_id=mapping_id,
                    source_scale_id=source_scale_id,
                    target_scale_id=target_scale_id,
                    source_value=source_value,
                    target_value=target_value,
                    mapping_confidence=confidence,
                )

                await upsert_grading_scale_mapping(mapping)

                mapping_stats["total_mappings"] += 1
                if confidence == "exact":
                    mapping_stats["exact_mappings"] += 1
                elif confidence == "approximate":
                    mapping_stats["approximate_mappings"] += 1
                else:
                    mapping_stats["manual_review_required"] += 1

                logger.debug(
                    f"Stored mapping: {source_scale}:{source_value} -> "
                    f"{target_scale}:{target_value} (confidence: {confidence})"
                )

    # ============================================
    # Summary
    # ============================================
    logger.info("=" * 60)
    logger.info("GRADING SCALE BOOTSTRAP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Registered scales: {len(registered_scales)}")
    logger.info(f"Total mappings: {mapping_stats['total_mappings']}")
    logger.info(f"  - Exact mappings: {mapping_stats['exact_mappings']}")
    logger.info(f"  - Approximate mappings: {mapping_stats['approximate_mappings']}")
    logger.info(
        f"  - Manual review required: {mapping_stats['manual_review_required']}"
    )
    logger.info("=" * 60)

    return mapping_stats
