DROP TRIGGER IF EXISTS set_label_from_original ON disease_grading;
DROP FUNCTION IF EXISTS trg_unpack_from_original_grade();
CREATE OR REPLACE FUNCTION trg_unpack_from_original_grade()
RETURNS TRIGGER AS $$
DECLARE
    v_labels JSONB;
BEGIN
    -- Only run when we actually have an original grade value
    IF NEW.original_grade IS NOT NULL THEN

        -- Get the JSON map for this scale
        SELECT value_labels
        INTO v_labels
        FROM grading_scales
        WHERE scale_id = NEW.scale_id;

        -- Look up using the RAW original_grade as key
        NEW.grade_label := v_labels ->> NEW.original_grade;

    ELSE
        NEW.grade_label := NULL;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER set_label_from_original
BEFORE INSERT OR UPDATE OF original_grade, scale_id
ON disease_grading
FOR EACH ROW
EXECUTE FUNCTION trg_unpack_from_original_grade();









