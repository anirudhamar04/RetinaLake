# Storage Architecture

RetinaLake uses a **hybrid storage approach** for images and masks.

## Original data (`STORAGE_DATA_ROOT`)

- **Never modified**: original dataset files remain untouched.
- **Source of truth**: all original images, masks, and annotations.
- **Read-only**: used for reference and provenance.

## Processed data (`STORAGE_LOCAL_ROOT` / `processed/`)

- **Transformed masks only**: written only when conversion/processing actually occurs.
- **Organized by dataset**: `processed/<dataset_name>/masks/<annotation_type>/`.
- **UUID-based naming**: `<first_8_chars_of_uuid>.png` for traceability.

## When masks are saved to `processed/`

| Scenario                          | Saved? | Example                         |
| --------------------------------- | ------ | ------------------------------- |
| Binary mask validation only       | ❌ No  | DRIVE vessel masks (used as-is) |
| Extract class from multi-class    | ✅ Yes | ORIGA disc extraction (class 1) |
| Contour → binary mask conversion  | ✅ Yes | Drishti-GS1 contours to masks   |
| XML → binary mask conversion      | ✅ Yes | ImageRet lesion polygons        |
| Soft map registration             | ❌ No  | LAG attention maps (used as-is) |

**Database fields:**

- `mask_file_path`: points to the mask to use (either original or processed).
- `original_file_path`: always points to the source in `STORAGE_DATA_ROOT`.

**Example:**

```
Contour conversion:
  mask_file_path: processed/Drishti-GS1/masks/optic_disc/a1b2c3d4.png (converted)
  original_file_path: data/Drishti-GS1/.../drishtiGS_031_ODAvgBoundary.txt (source)

Validation only:
  mask_file_path: data/DRIVE/masks/01_manual1.gif (original)
  original_file_path: data/DRIVE/masks/01_manual1.gif (same)
```

See the [Ingestion Framework guide](ingestion_framework.md) for detailed documentation on mask
conversion and the segmentation processor, and the [Schema Reference](schema_reference.md) for
the full set of storage-locator columns on `images`.
