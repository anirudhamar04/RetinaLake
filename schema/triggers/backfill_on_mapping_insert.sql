-- ============================================
-- Trigger: Backfill Grades on New Mapping
-- ============================================
--
-- This trigger automatically updates existing disease_grading records when
-- a new scale mapping is added to the grading_scale_mappings table.
--
-- How it works:
-- 1. When a new mapping is inserted into grading_scale_mappings
-- 2. Find all disease_grading records that:
--    - Use the source_scale_id
--    - Have original_grade = source_value
--    - Have scaled_grade IS NULL (no previous conversion)
-- 3. Update their scaled_grade to the target_value
--
-- This allows incremental learning of new scales:
-- - Ingest data with unknown scale → scaled_grade = NULL
-- - Later, add mapping to grading_scale_mappings
-- - Trigger automatically backfills all historical records
--
-- Note: Only updates records with scaled_grade IS NULL to preserve manual overrides.
--
-- Audit: the UPDATE below fires the BEFORE UPDATE auto_convert_disease_grade trigger
-- for each affected row, which emits a per-row 'grade_conversion' NOTIFY carrying that
-- row's own provenance_chain_id. The Python listener records those idempotently, so this
-- trigger no longer writes its own transformation_operations row (the old aggregate
-- gen_random_uuid() insert could not be linked to per-row provenance chains).
-- ============================================

CREATE OR REPLACE FUNCTION backfill_grades_on_new_mapping()
RETURNS TRIGGER AS $$
DECLARE
    affected_count INTEGER;
    target_scale_name TEXT;
    affected_grading_ids UUID[];
BEGIN
    -- Get the target scale name for logging
    SELECT scale_name INTO target_scale_name
    FROM grading_scales
    WHERE scale_id = NEW.target_scale_id;

    -- Update all disease_grading records that match this mapping
    -- Only update records where scaled_grade IS NULL (avoid overwriting existing conversions)
    WITH updated AS (
        UPDATE disease_grading
        SET 
            scaled_grade = NEW.target_value,
            updated_at = now()
        WHERE scale_id = NEW.source_scale_id
          AND original_grade = NEW.source_value
          AND scaled_grade IS NULL
        RETURNING grading_id
    )
    SELECT COUNT(*), array_agg(grading_id) INTO affected_count, affected_grading_ids
    FROM updated;

    IF affected_count > 0 THEN
        RAISE NOTICE 'Backfilled % disease_grading records: scale % value % -> % (target scale: %)',
                     affected_count,
                     NEW.source_scale_id,
                     NEW.source_value,
                     NEW.target_value,
                     target_scale_name;
        -- Audit rows are emitted per-row by the BEFORE UPDATE auto_convert_disease_grade
        -- trigger (fired by the UPDATE above) and recorded by the Python listener.
    ELSE
        RAISE DEBUG 'No disease_grading records needed backfill for scale % value %',
                    NEW.source_scale_id,
                    NEW.source_value;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop trigger if exists (for idempotency)
DROP TRIGGER IF EXISTS trigger_backfill_grades_on_new_mapping ON grading_scale_mappings;

-- Create trigger
CREATE TRIGGER trigger_backfill_grades_on_new_mapping
    AFTER INSERT ON grading_scale_mappings
    FOR EACH ROW
    EXECUTE FUNCTION backfill_grades_on_new_mapping();

-- ============================================
-- Comments
-- ============================================
COMMENT ON FUNCTION backfill_grades_on_new_mapping() IS 
'Automatically backfills disease_grading.scaled_grade when a new mapping is added to grading_scale_mappings.
Only updates records where scaled_grade IS NULL to preserve manual overrides.
Enables incremental learning of new grading scales.';

COMMENT ON TRIGGER trigger_backfill_grades_on_new_mapping ON grading_scale_mappings IS 
'Backfills historical disease grades when new scale mappings are learned';
