"""
Comprehensive tests for bootstrap_scale_mappings using real SUSTech-SYSU data.

Tests cover:
- Scale registration
- Mapping analysis from real drLabels.csv
- Mapping validation and confidence scoring
- Complete bootstrap process
"""

import pytest
from pathlib import Path

from chaksudb.config.config import get_data_root
from chaksudb.db.queries.grading import find_grading_scale_by_id
from chaksudb.ingest.framework.gen_uuid import generate_grading_scale_uuid
from chaksudb.ingest.framework.scale_bootstrap.bootstrap_scale_mappings import (
    SCALE_DEFINITIONS,
    analyze_scale_mappings,
    validate_mappings,
    bootstrap_grading_scales,
)


pytestmark = pytest.mark.asyncio


class TestScaleBootstrapWithRealData:
    """Test scale bootstrap with real SUSTech-SYSU data."""

    @pytest.fixture
    def sustech_data_dir(self):
        """Get path to SUSTech-SYSU dataset."""
        data_root = get_data_root()
        return data_root / "30_SUSTech-SYSU"

    @pytest.fixture
    def dr_labels_path(self, sustech_data_dir):
        """Get path to drLabels.csv."""
        return sustech_data_dir / "drLabels.csv"

    async def test_scale_definitions_completeness(self):
        """Test that all required scale definitions are present."""
        required_scales = ["ICDR_0_4", "ICDR_0_5", "AAO", "Scottish"]
        
        for scale_name in required_scales:
            assert scale_name in SCALE_DEFINITIONS
            
            scale_def = SCALE_DEFINITIONS[scale_name]
            assert "scale_name" in scale_def
            assert "disease_type" in scale_def
            assert "scale_description" in scale_def
            assert "min_value" in scale_def
            assert "max_value" in scale_def
            assert "value_labels" in scale_def

    async def test_scale_definitions_value_labels(self):
        """Test that scale definitions have appropriate value labels."""
        icdr_0_4 = SCALE_DEFINITIONS["ICDR_0_4"]
        assert icdr_0_4["disease_type"] == "DR"
        assert icdr_0_4["min_value"] == 0
        assert icdr_0_4["max_value"] == 4
        assert len(icdr_0_4["value_labels"]) == 5  # 0, 1, 2, 3, 4
        assert "0" in icdr_0_4["value_labels"]
        assert "4" in icdr_0_4["value_labels"]

        icdr_0_5 = SCALE_DEFINITIONS["ICDR_0_5"]
        assert icdr_0_5["max_value"] == 5
        assert len(icdr_0_5["value_labels"]) == 6  # 0, 1, 2, 3, 4, 5
        assert "5" in icdr_0_5["value_labels"]

    async def test_analyze_scale_mappings_real_data(self, dr_labels_path):
        """Test analyzing scale mappings from real drLabels.csv."""
        if not dr_labels_path.exists():
            pytest.skip(f"SUSTech-SYSU drLabels.csv not found at {dr_labels_path}")

        mapping_observations = analyze_scale_mappings(dr_labels_path)

        # Check that we got mappings for all expected pairs
        assert "ICDR_0_4" in mapping_observations
        assert "AAO" in mapping_observations
        assert "Scottish" in mapping_observations

        # Check ICDR -> AAO mappings
        assert "AAO" in mapping_observations["ICDR_0_4"]
        icdr_to_aao = mapping_observations["ICDR_0_4"]["AAO"]
        
        # Should have mappings for grades 0-4 at minimum
        assert "0" in icdr_to_aao
        assert len(icdr_to_aao["0"]) > 0  # Has observations

        # Check Scottish -> ICDR mappings
        assert "ICDR_0_4" in mapping_observations["Scottish"]

    async def test_analyze_scale_mappings_bidirectional(self, dr_labels_path):
        """Test that mappings are bidirectional."""
        if not dr_labels_path.exists():
            pytest.skip(f"SUSTech-SYSU drLabels.csv not found at {dr_labels_path}")

        mapping_observations = analyze_scale_mappings(dr_labels_path)

        # Check that we have both ICDR->AAO and AAO->ICDR
        assert "AAO" in mapping_observations["ICDR_0_4"]
        assert "ICDR_0_4" in mapping_observations["AAO"]

        # Check that we have both ICDR->Scottish and Scottish->ICDR
        assert "Scottish" in mapping_observations["ICDR_0_4"]
        assert "ICDR_0_4" in mapping_observations["Scottish"]

        # Check that we have AAO<->Scottish
        assert "Scottish" in mapping_observations["AAO"]
        assert "AAO" in mapping_observations["Scottish"]

    async def test_validate_mappings_consistency(self, dr_labels_path):
        """Test mapping validation produces consistent results."""
        if not dr_labels_path.exists():
            pytest.skip(f"SUSTech-SYSU drLabels.csv not found at {dr_labels_path}")

        mapping_observations = analyze_scale_mappings(dr_labels_path)
        validated_mappings = validate_mappings(mapping_observations)

        # Check that validated mappings have the expected structure
        assert isinstance(validated_mappings, dict)

        for source_scale in validated_mappings:
            for target_scale in validated_mappings[source_scale]:
                for source_value, (target_value, confidence) in validated_mappings[source_scale][target_scale].items():
                    # Check confidence is one of the expected values
                    assert confidence in ["exact", "approximate", "manual_review_required"]
                    
                    # Check that target_value is an integer
                    assert isinstance(target_value, int)
                    
                    # Check that source_value is a string
                    assert isinstance(source_value, str)

    async def test_validate_mappings_confidence_levels(self, dr_labels_path):
        """Test that confidence levels are assigned appropriately."""
        if not dr_labels_path.exists():
            pytest.skip(f"SUSTech-SYSU drLabels.csv not found at {dr_labels_path}")

        mapping_observations = analyze_scale_mappings(dr_labels_path)
        validated_mappings = validate_mappings(mapping_observations)

        # Collect confidence distribution
        confidence_counts = {"exact": 0, "approximate": 0, "manual_review_required": 0}

        for source_scale in validated_mappings:
            for target_scale in validated_mappings[source_scale]:
                for _, (_, confidence) in validated_mappings[source_scale][target_scale].items():
                    confidence_counts[confidence] += 1

        # We expect most mappings to be exact or approximate
        assert confidence_counts["exact"] > 0
        # At least some mappings should be marked as exact
        print(f"Confidence distribution: {confidence_counts}")

    async def test_bootstrap_grading_scales_real_data(self, sustech_data_dir):
        """Test complete bootstrap process with real data."""
        if not sustech_data_dir.exists():
            pytest.skip(f"SUSTech-SYSU directory not found at {sustech_data_dir}")

        # Run bootstrap
        stats = await bootstrap_grading_scales(
            sustech_data_dir=sustech_data_dir,
            force_reload=False,
        )

        # Check summary statistics
        assert "total_mappings" in stats
        assert "exact_mappings" in stats
        assert "approximate_mappings" in stats
        assert "manual_review_required" in stats

        # Should have created some mappings
        assert stats["total_mappings"] > 0

        print(f"Bootstrap statistics: {stats}")

    async def test_bootstrap_registers_all_scales(self, sustech_data_dir):
        """Test that bootstrap registers all expected scales."""
        if not sustech_data_dir.exists():
            pytest.skip(f"SUSTech-SYSU directory not found at {sustech_data_dir}")

        await bootstrap_grading_scales(
            sustech_data_dir=sustech_data_dir,
            force_reload=False,
        )

        # Check that all scales were registered
        for scale_key, scale_def in SCALE_DEFINITIONS.items():
            scale_id = generate_grading_scale_uuid(
                scale_def["scale_name"],
                scale_def["disease_type"],
            )
            
            scale = await find_grading_scale_by_id(scale_id)
            assert scale is not None
            assert scale.scale_name == scale_def["scale_name"]
            assert scale.disease_type == scale_def["disease_type"]

    async def test_bootstrap_idempotency(self, sustech_data_dir):
        """Test that running bootstrap twice produces the same result."""
        if not sustech_data_dir.exists():
            pytest.skip(f"SUSTech-SYSU directory not found at {sustech_data_dir}")

        # Run bootstrap first time
        stats_1 = await bootstrap_grading_scales(
            sustech_data_dir=sustech_data_dir,
            force_reload=False,
        )

        # Run bootstrap second time
        stats_2 = await bootstrap_grading_scales(
            sustech_data_dir=sustech_data_dir,
            force_reload=False,
        )

        # Statistics should be similar (may vary slightly due to upserts)
        assert stats_1["total_mappings"] == stats_2["total_mappings"]

    async def test_bootstrap_handles_missing_file(self):
        """Test that bootstrap handles missing files gracefully."""
        nonexistent_dir = Path("/nonexistent/directory")

        with pytest.raises(FileNotFoundError):
            await bootstrap_grading_scales(sustech_data_dir=nonexistent_dir)

    async def test_icdr_0_4_has_correct_labels(self):
        """Test that ICDR_0_4 scale has correct labels."""
        icdr_0_4 = SCALE_DEFINITIONS["ICDR_0_4"]
        
        expected_labels = {
            "0": "No DR",
            "1": "Mild NPDR",
            "2": "Moderate NPDR",
            "3": "Severe NPDR",
            "4": "PDR (Proliferative DR)",
        }

        assert icdr_0_4["value_labels"] == expected_labels

    async def test_icdr_0_5_includes_ungradable(self):
        """Test that ICDR_0_5 scale includes ungradable grade 5."""
        icdr_0_5 = SCALE_DEFINITIONS["ICDR_0_5"]
        
        assert icdr_0_5["max_value"] == 5
        assert "5" in icdr_0_5["value_labels"]
        assert "ungradable" in icdr_0_5["value_labels"]["5"].lower()

    async def test_aao_scale_definition(self):
        """Test AAO scale definition."""
        aao = SCALE_DEFINITIONS["AAO"]
        
        assert aao["disease_type"] == "DR"
        assert aao["min_value"] == 0
        assert aao["max_value"] == 5
        assert len(aao["value_labels"]) == 6

    async def test_scottish_scale_definition(self):
        """Test Scottish scale definition."""
        scottish = SCALE_DEFINITIONS["Scottish"]
        
        assert scottish["disease_type"] == "DR"
        assert scottish["min_value"] == 0
        assert scottish["max_value"] == 5
        assert len(scottish["value_labels"]) == 6

    async def test_mapping_observations_grade_0_consistency(self, dr_labels_path):
        """Test that grade 0 (no DR) maps consistently across scales."""
        if not dr_labels_path.exists():
            pytest.skip(f"SUSTech-SYSU drLabels.csv not found at {dr_labels_path}")

        mapping_observations = analyze_scale_mappings(dr_labels_path)
        validated_mappings = validate_mappings(mapping_observations)

        # Grade 0 should map to grade 0 in all scales
        if "ICDR_0_4" in validated_mappings and "AAO" in validated_mappings["ICDR_0_4"]:
            if "0" in validated_mappings["ICDR_0_4"]["AAO"]:
                target_value, confidence = validated_mappings["ICDR_0_4"]["AAO"]["0"]
                assert target_value == 0  # No DR should map to No DR
                assert confidence == "exact"  # Should be exact mapping

    async def test_mapping_observations_grade_4_pdr(self, dr_labels_path):
        """Test that grade 4 (PDR) maps consistently across scales."""
        if not dr_labels_path.exists():
            pytest.skip(f"SUSTech-SYSU drLabels.csv not found at {dr_labels_path}")

        mapping_observations = analyze_scale_mappings(dr_labels_path)
        validated_mappings = validate_mappings(mapping_observations)

        # Grade 4 (PDR) should map to grade 4 in AAO
        if "ICDR_0_4" in validated_mappings and "AAO" in validated_mappings["ICDR_0_4"]:
            if "4" in validated_mappings["ICDR_0_4"]["AAO"]:
                target_value, confidence = validated_mappings["ICDR_0_4"]["AAO"]["4"]
                assert target_value == 4  # PDR should map to PDR
                assert confidence in ["exact", "approximate"]

    async def test_analyze_scale_mappings_handles_missing_data(self):
        """Test that analyze handles missing or invalid data gracefully."""
        # Create a temporary CSV with incomplete data
        import tempfile
        import csv

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow([
                "Fundus_images",
                "left_versus_right_eye(left_0_right_1)",
                "DR_grade(International_Clinical_DR_Severity_Scale)",
                "DR_grade(American_Academy_of_Ophthalmology)",
                "DR_grade(Scottish_DR_grading_protocol)"
            ])
            # Write a row with invalid data
            writer.writerow(["test.jpg", "0", "invalid", "2", "2"])
            # Write a row with missing data
            writer.writerow(["test2.jpg", "0", "1", "", "1"])
            
            temp_path = Path(f.name)

        try:
            # Should handle gracefully without crashing
            mapping_observations = analyze_scale_mappings(temp_path)
            # May have some mappings or may be empty due to invalid data
            assert isinstance(mapping_observations, dict)
        finally:
            temp_path.unlink()  # Clean up temp file
