-- ============================================
-- Trigger: Automatic Grade Conversion
-- ============================================
-- 
-- This trigger automatically converts disease grades to the target scale (ICDR_0_4)
-- whenever a new grading record is inserted or updated.
--
-- How it works:
-- 1. On INSERT or UPDATE of disease_grading table
-- 2. If the scale_id is different from the target scale (ICDR_0_4)
-- 3. Look up the mapping in grading_scale_mappings
-- 4. Automatically populate scaled_grade with the converted value
-- 5. If no mapping exists, leave scaled_grade as NULL
--
-- This allows datasets to be ingested with their native scales, and conversion
-- happens automatically in the database.
--
-- Each conversion emits a 'grade_conversion' NOTIFY event (payload carries the
-- grading_id, scaled_grade, provenance_chain_id, etc.). A Python listener
-- (chaksudb/ingest/framework/provenance_listener.py) consumes these and records
-- the audit row in transformation_operations + provenance_transformations using
-- deterministic, idempotent UUIDs. The trigger itself NO LONGER writes audit rows
-- (the old gen_random_uuid() inserts duplicated on every idempotent re-upsert).
-- A reconciliation sweep (reconcile_grade_conversions) is the completeness backstop
-- for events dropped while no listener was connected.
-- ============================================

CREATE OR REPLACE FUNCTION auto_convert_disease_grade()
RETURNS TRIGGER AS $$
DECLARE
    v_target_scale_id UUID;
    target_scale_name TEXT := 'ICDR_0_4';  -- Default target scale
    mapped_value INTEGER;
    did_convert BOOLEAN := FALSE;
BEGIN
    -- Get the target scale ID for this disease type
    SELECT scale_id INTO v_target_scale_id
    FROM grading_scales
    WHERE scale_name = target_scale_name
      AND disease_type = NEW.disease_type;

    -- If target scale doesn't exist, log warning and continue
    IF v_target_scale_id IS NULL THEN
        RAISE WARNING 'Target scale % not found for disease type %', 
                      target_scale_name, NEW.disease_type;
        RETURN NEW;
    END IF;

    -- If the incoming scale is already the target scale, set scaled_grade = original_grade
    IF NEW.scale_id = v_target_scale_id THEN
        -- Direct assignment: scaled_grade = original_grade (as integer)
        IF NEW.original_grade IS NOT NULL THEN
            BEGIN
                NEW.scaled_grade := NEW.original_grade::INTEGER;
                did_convert := TRUE;
            EXCEPTION WHEN OTHERS THEN
                RAISE WARNING 'Could not convert original_grade % to integer for grading_id %',
                              NEW.original_grade, NEW.grading_id;
                NEW.scaled_grade := NULL;
            END;
        END IF;
        IF did_convert AND NEW.provenance_chain_id IS NOT NULL THEN
            PERFORM pg_notify(
                'grade_conversion',
                jsonb_build_object(
                    'mode', 'same_scale',
                    'grading_id', NEW.grading_id,
                    'image_id', NEW.image_id,
                    'scale_id', NEW.scale_id,
                    'original_grade', NEW.original_grade,
                    'disease_type', NEW.disease_type,
                    'scaled_grade', NEW.scaled_grade,
                    'target_scale_id', v_target_scale_id,
                    'target_scale_name', target_scale_name,
                    'provenance_chain_id', NEW.provenance_chain_id
                )::text
            );
        END IF;
        NEW.updated_at := now();
        RETURN NEW;
    END IF;

    -- Otherwise, look up the mapping
    IF NEW.original_grade IS NOT NULL THEN
        SELECT target_value INTO mapped_value
        FROM grading_scale_mappings
        WHERE source_scale_id = NEW.scale_id
          AND target_scale_id = v_target_scale_id
          AND source_value = NEW.original_grade
        LIMIT 1;

        IF mapped_value IS NOT NULL THEN
            NEW.scaled_grade := mapped_value;
            did_convert := TRUE;
            RAISE DEBUG 'Auto-converted grade: % (scale %) -> % (scale %)',
                        NEW.original_grade, NEW.scale_id,
                        mapped_value, v_target_scale_id;
        ELSE
            -- No mapping found - leave scaled_grade as NULL
            NEW.scaled_grade := NULL;
            RAISE WARNING 'No mapping found for scale % value % to target scale %. scaled_grade set to NULL.',
                          NEW.scale_id, NEW.original_grade, target_scale_name;
        END IF;
    ELSE
        -- original_grade is NULL, can't convert
        NEW.scaled_grade := NULL;
    END IF;

    -- Emit a NOTIFY event when we performed a conversion; the Python listener
    -- records the audit row idempotently. Skip when there is no provenance chain
    -- to link the transformation to.
    IF did_convert AND NEW.provenance_chain_id IS NOT NULL THEN
        PERFORM pg_notify(
            'grade_conversion',
            jsonb_build_object(
                'mode', 'mapped',
                'grading_id', NEW.grading_id,
                'image_id', NEW.image_id,
                'scale_id', NEW.scale_id,
                'original_grade', NEW.original_grade,
                'disease_type', NEW.disease_type,
                'scaled_grade', NEW.scaled_grade,
                'target_scale_id', v_target_scale_id,
                'target_scale_name', target_scale_name,
                'provenance_chain_id', NEW.provenance_chain_id
            )::text
        );
    END IF;

    -- Set updated_at timestamp
    NEW.updated_at := now();

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop trigger if exists (for idempotency)
DROP TRIGGER IF EXISTS trigger_auto_convert_disease_grade ON disease_grading;

-- Create trigger
CREATE TRIGGER trigger_auto_convert_disease_grade
    BEFORE INSERT OR UPDATE ON disease_grading
    FOR EACH ROW
    EXECUTE FUNCTION auto_convert_disease_grade();

-- ============================================
-- Comments
-- ============================================
COMMENT ON FUNCTION auto_convert_disease_grade() IS 
'Automatically converts disease grades to target scale (ICDR_0_4) using grading_scale_mappings table. 
If scale is already target scale, sets scaled_grade = original_grade. 
If no mapping exists, leaves scaled_grade as NULL.';

COMMENT ON TRIGGER trigger_auto_convert_disease_grade ON disease_grading IS 
'Automatically converts grades to standard scale on INSERT/UPDATE';
