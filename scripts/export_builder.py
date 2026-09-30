"""
ExportSpec Builder — interactive Gradio UI that generates ready-to-run Python
code for any combination of ChaksuDB filters.

Two tabs:
  • Easy      — pick modality / datasets / task (DR·AMD·Glaucoma classification,
                OD·OC·vessels·AV·lesions segmentation, disease grading, normal/abnormal),
                or "flatten everything" for a dataset. No dataset = all datasets.
  • Advanced  — every ExportSpec knob, including the newer ones (disease concepts,
                positive-for filter, health status, quality-type pivot).

Run:
    uv run python scripts/export_builder.py
    uv run python scripts/export_builder.py --port 7860
    uv run python scripts/export_builder.py --port 7860 --share   # public tunnel
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from chaksudb.export.spec import ExportSpec
from chaksudb.export.streaming import count_rows
log = logging.getLogger("export_builder")

# ── static option lists ───────────────────────────────────────────────────────
# Each entry is (display label, internal value). Gradio returns the internal
# value to all callbacks, so generate_code / preview_count need no changes.

ANNOTATION_TASKS = [
    ("Disease grading",                               "grading"),
    ("Segmentation masks",                            "segmentation"),
    ("Classification labels",                         "classification"),
    ("Object locations (bounding boxes / keypoints)", "localization"),
    ("Image quality scores",                          "quality"),
    ("Clinical keywords",                             "keyword"),
    ("Clinical descriptions",                         "description"),
]
MODALITIES = [
    ("Fundus photography",       "fundus"),
    ("OCT scan",                 "oct"),
    ("Ultra-widefield (UWF)",    "uwf"),
    ("Fluorescein angiography",  "fa"),
]
SPLITS        = ["train", "val", "test"]
DISEASE_TYPES = ["DR", "DME", "Glaucoma", "AMD", "myopic_maculopathy"]
SEG_TYPES = [
    ("Vessels",                       "vessels"),
    ("Optic disc",                    "optic_disc"),
    ("Optic cup",                     "optic_cup"),
    ("Lesions",                       "lesions"),
    ("Fovea",                         "fovea"),
    ("Choroid",                       "choroid"),
    ("Retinal layers",                "retinal_layer"),
    ("Nerve fiber layer",             "nerve_fiber"),
    ("Artery/vein color mask (RGB)",  "av"),
]
LESION_SUBTYPES = [
    ("Microaneurysms",              "microaneurysm"),
    ("Hemorrhages",                 "hemorrhage"),
    ("Hard exudates",               "hard_exudate"),
    ("Soft exudates / cotton wool", "soft_exudate"),
    ("Neovascularization",          "neovascularization"),
    ("Drusen",                      "drusen"),
]
LOC_TYPES = [
    ("Bounding box",  "bounding_box"),
    ("Keypoint",      "keypoint"),
    ("Center point",  "center_point"),
    ("Polygon",       "polygon"),
    ("Circle",        "circle"),
]
IQA_LABELS = [
    ("Good",       "good"),
    ("Usable",     "usable"),
    ("Poor / bad", "bad"),
]
SPLIT_TYPES = [
    ("Any (no filter)",                              "(any)"),
    ("Manually assigned splits (includes val set)",  "user_defined"),
    ("Explicitly labeled",                           "explicit"),
    ("Metadata-defined",                             "metadata_defined"),
]
ANNOTATION_SOURCES = [
    ("Best available — consensus if exists, otherwise expert", "prefer_consensus"),
    ("Expert annotations only",                                "expert_only"),
    ("Consensus annotations only",                             "consensus_only"),
    ("Both expert and consensus rows",                         "both"),
]
CAPTION_MODES = [
    ("None",                  "(none)"),
    ("Clinical summary",      "clinical"),
    ("Keywords",              "keyword"),
    ("Grading summary",       "grading"),
    ("Classification summary","classification"),
    ("Synthetic caption",     "synthetic"),
    ("All caption types",     "all"),
]
OUTPUT_FORMATS = [
    ("None",                "(none)"),
    ("Classification",      "classification"),
    ("Grading",             "grading"),
    ("Segmentation",        "segmentation"),
    ("Object detection",    "detection"),
    ("Vision-language",     "vision_language"),
    ("Self-supervised (SSL)","ssl"),
]
DETECTION_FORMATS = [
    ("Standard list (default)", "nested"),
    ("COCO format",             "coco"),
]
LABEL_TYPES = [
    ("Integer class index  (0, 1, 2…)", "int"),
    ("Float / probability",             "float"),
]
REQUIRE_MODES = [
    ("All images, even those without labels",       "none"),
    ("Only images that have ALL selected labels",   "all"),
    ("Only images that have at least one label",    "any"),
]
# Easy-mode: only the two modes people actually reach for.
EASY_REQUIRE_MODES = [
    ("Only images that have at least one of the picked labels", "any"),
    ("Only images that have ALL of the picked labels",         "all"),
]

# ── newer knobs ────────────────────────────────────────────────────────────────
# Canonical disease concepts: unified per-disease columns/filters that work across
# binary / multi_label / multi_class storage in any dataset.
CONCEPTS = [
    ("Diabetic Retinopathy (DR)", "DR"),
    ("AMD",                        "AMD"),
    ("Glaucoma",                   "Glaucoma"),
    ("DME",                        "DME"),
    ("Cataract",                   "cataract"),
]
# Easy-mode segmentation choices (OD / OC / vessels / AV / lesions).
EASY_SEG_TYPES = [
    ("Optic disc (OD)",     "optic_disc"),
    ("Optic cup (OC)",      "optic_cup"),
    ("Vessels",             "vessels"),
    ("Artery/Vein (AV)",    "av"),
    ("Lesions",             "lesions"),
]
HEALTH_FILTERS = [
    ("No health filter",            "(none)"),
    ("Only NORMAL images",          "normal"),
    ("Only DISEASED (abnormal)",    "abnormal"),
]
QUALITY_TYPE_OPTS = [
    ("Overall",          "overall"),
    ("Gradability",      "gradability"),
    ("Clarity / focus",  "clarity"),
    ("Field definition", "field_definition"),
    ("Artifact",         "artifact"),
    ("Contrast",         "contrast"),
    ("Blur",             "blur"),
    ("Illumination",     "illumination"),
]


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _query_dataset_modalities() -> tuple[list[str], dict[str, list[str]]]:
    """
    Query only datasets and their modalities from the live DB.
    Returns (dataset_names, {dataset_name: [modality, ...]}).
    Called on every modality-filter change so the list is always current.
    """
    from chaksudb.db.connection import execute_query

    dataset_rows = await execute_query(
        "SELECT dataset_name FROM datasets ORDER BY dataset_name"
    )
    datasets = [r[0] for r in dataset_rows]

    mod_rows = await execute_query(
        """
        SELECT d.dataset_name, i.modality
        FROM images i
        JOIN datasets d ON i.dataset_id = d.dataset_id
        WHERE i.modality IS NOT NULL
        GROUP BY d.dataset_name, i.modality
        ORDER BY d.dataset_name
        """
    )
    dm: dict[str, set] = {}
    for dataset_name, modality in mod_rows:
        dm.setdefault(dataset_name, set()).add(modality)
    return datasets, {k: list(v) for k, v in dm.items()}


async def _query_db_options() -> dict:
    """
    Query the live DB for all dynamic option lists (used once at startup).
    Returns a dict with keys: datasets, dataset_modalities, class_names, db_error.
    Falls back to empty lists on any connection error.
    """
    from chaksudb.db.connection import execute_query, close_pool

    results = {
        "datasets": [],
        "dataset_modalities": {},   # {dataset_name: [modality, ...]}
        "class_names": [],
        "db_error": None,
    }
    try:
        results["datasets"], results["dataset_modalities"] = (
            await _query_dataset_modalities()
        )
        class_rows = await execute_query(
            "SELECT DISTINCT class_name FROM classification_annotations ORDER BY class_name"
        )
        results["class_names"] = [r[0] for r in class_rows]
    except Exception as e:
        results["db_error"] = str(e)
        log.warning("Could not reach DB for option lists: %s", e)
    finally:
        try:
            await close_pool()
        except Exception:
            pass
    return results


def fetch_db_options() -> dict:
    """Synchronous wrapper around the async DB query."""
    try:
        return asyncio.run(_query_db_options())
    except Exception as e:
        return {"datasets": [], "db_error": str(e)}


def _datasets_for_modalities(selected_modalities, all_datasets, dataset_modalities):
    """Return datasets that have at least one of the selected modalities."""
    if not selected_modalities:
        return all_datasets
    return [
        ds for ds in all_datasets
        if any(m in dataset_modalities.get(ds, []) for m in selected_modalities)
    ]


# ── code generator ────────────────────────────────────────────────────────────

def _repr(v):
    """Python repr for a value, producing clean single-line output."""
    if isinstance(v, list):
        return "[" + ", ".join(json.dumps(x) for x in v) + "]"
    if isinstance(v, dict):
        return json.dumps(v)
    if isinstance(v, str):
        return json.dumps(v)
    return repr(v)


def _render_spec_code(fields: dict, out_path: str, warnings=None) -> str:
    """Render an ExportSpec(...) + export() snippet from a field dict."""
    lines = [
        "from chaksudb.export.spec import ExportSpec",
        "from chaksudb.export.api import export",
        "",
    ]
    if warnings:
        lines.extend(warnings)
        lines.append("")
    if not fields:
        lines.append("spec = ExportSpec()  # no filters — returns all images")
    else:
        lines.append("spec = ExportSpec(")
        for k, v in fields.items():
            lines.append(f"    {k}={_repr(v)},")
        lines.append(")")
    lines += ["", f'export(spec, parquet_path="{out_path or "out.parquet"}")']
    return "\n".join(lines)


def _assemble_advanced_fields(d: dict) -> tuple[dict, list[str]]:
    """Turn the Advanced-tab UI state (a dict) into ExportSpec field kwargs + warnings.

    Shared by both the code generator and the row-count preview so they never drift.
    """
    fields: dict = {}
    warnings: list[str] = []

    # datasets & splits
    if d.get("dataset_names"):
        fields["dataset_names"] = d["dataset_names"]
    if d.get("split_names"):
        fields["split_names"] = d["split_names"]
    if d.get("split_type") and d["split_type"] != "(any)":
        fields["split_type"] = d["split_type"]

    # image scope
    if d.get("modalities"):
        fields["modalities"] = d["modalities"]

    # annotation tasks (concepts/positive_for imply classification; quality_types imply quality)
    tasks = list(d.get("annotation_tasks") or [])
    if (d.get("classification_concepts") or d.get("classification_positive_for")) and "classification" not in tasks:
        tasks.append("classification")
    if d.get("quality_types") and "quality" not in tasks:
        tasks.append("quality")
    if tasks:
        fields["annotation_tasks"] = tasks
    if d.get("require_annotations_mode") and d["require_annotations_mode"] != "none":
        fields["require_annotations_mode"] = d["require_annotations_mode"]
    if d.get("annotation_source") and d["annotation_source"] != "prefer_consensus":
        fields["annotation_source"] = d["annotation_source"]

    # grading
    if "grading" in tasks:
        if d.get("disease_types"):
            fields["disease_types"] = d["disease_types"]
        if d.get("include_original_grade") is False:
            fields["include_original_grade"] = False
        if d.get("include_scaled_grade") is False:
            fields["include_scaled_grade"] = False
        gf = d.get("grade_filter_json")
        if gf and gf.strip():
            try:
                fields["grade_filter"] = json.loads(gf)
            except json.JSONDecodeError as e:
                warnings.append(f"# ⚠  grade_filter JSON is invalid: {e}")

    # segmentation
    if "segmentation" in tasks:
        if d.get("segmentation_types"):
            fields["segmentation_types"] = d["segmentation_types"]
        if d.get("lesion_subtypes") and "lesions" in (d.get("segmentation_types") or []):
            fields["lesion_subtypes"] = d["lesion_subtypes"]

    # classification — concepts (unified per-disease) and/or exact class_name pivots
    if "classification" in tasks:
        if d.get("classification_concepts"):
            fields["classification_concepts"] = d["classification_concepts"]
        if d.get("classification_positive_for"):
            fields["classification_positive_for"] = d["classification_positive_for"]
        class_names = [s for s in (d.get("classification_class_names_raw") or []) if s]
        if class_names:
            fields["classification_class_names"] = class_names
        if not class_names and not d.get("classification_concepts"):
            warnings.append(
                "# ⚠  Pick at least one disease concept or class to get classification columns."
            )
        if d.get("classification_label_type") and d["classification_label_type"] != "int":
            fields["classification_label_type"] = d["classification_label_type"]

    # localization
    if "localization" in tasks:
        if d.get("localization_types"):
            fields["localization_types"] = d["localization_types"]
        if d.get("detection_format") and d["detection_format"] != "nested":
            fields["detection_format"] = d["detection_format"]

    # quality / IQA
    if d.get("quality_types"):
        fields["quality_types"] = d["quality_types"]
    if d.get("iqa_min_score_enabled"):
        fields["iqa_min_quality_score"] = round(d.get("iqa_min_quality_score", 0.0), 2)
    if d.get("iqa_quality_labels"):
        fields["iqa_quality_labels"] = d["iqa_quality_labels"]
    if d.get("include_fundus_roi"):
        fields["include_fundus_roi"] = True

    # health status (normal / abnormal)
    hsf = d.get("health_status_filter")
    if hsf and hsf != "(none)":
        fields["health_status_filter"] = hsf
        fields["include_health_status"] = True
    elif d.get("include_health_status"):
        fields["include_health_status"] = True

    # extras
    if d.get("include_patient_data"):
        fields["include_patient_data"] = True
    if d.get("caption_mode") and d["caption_mode"] != "(none)":
        fields["caption_mode"] = d["caption_mode"]
    if d.get("output_format") and d["output_format"] != "(none)":
        fields["output_format"] = d["output_format"]
    if d.get("base_path_for_paths") and d["base_path_for_paths"].strip():
        fields["base_path_for_paths"] = d["base_path_for_paths"].strip()

    return fields, warnings


# Advanced-tab input order (must match `advanced_inputs` in build_ui).
_ADVANCED_KEYS = [
    "dataset_names", "split_names", "split_type",
    "modalities",
    "annotation_tasks", "require_annotations_mode", "annotation_source",
    "disease_types", "include_original_grade", "include_scaled_grade", "grade_filter_json",
    "segmentation_types", "lesion_subtypes",
    "classification_class_names_raw", "classification_label_type",
    "localization_types", "detection_format",
    "iqa_min_score_enabled", "iqa_min_quality_score",
    "iqa_quality_labels", "include_fundus_roi",
    "include_patient_data", "caption_mode", "output_format",
    "base_path_for_paths",
    "parquet_path_str",
    # newer knobs (appended)
    "classification_concepts", "classification_positive_for", "quality_types",
    "include_health_status", "health_status_filter",
]


def generate_code(*args) -> str:
    """Advanced tab: build ExportSpec(...) + export() code from the full UI state."""
    d = dict(zip(_ADVANCED_KEYS, args))
    fields, warnings = _assemble_advanced_fields(d)
    return _render_spec_code(fields, (d.get("parquet_path_str") or "out.parquet").strip(), warnings)


async def preview_count(*args) -> str:
    """Advanced tab: build ExportSpec from current UI state and return the image count."""
    d = dict(zip(_ADVANCED_KEYS, args))
    fields, _ = _assemble_advanced_fields(d)
    try:
        n = await count_rows(ExportSpec(**fields))
        return f"**{n:,} images** match your current filters."
    except Exception as e:
        return f"**Could not count images:** {e}"


# ── easy-mode generator ─────────────────────────────────────────────────────────

def _assemble_easy_fields(d: dict) -> dict:
    """Turn Easy-tab choices into ExportSpec field kwargs (the flatten path is separate)."""
    fields: dict = {}
    if d.get("datasets"):
        fields["dataset_names"] = d["datasets"]
    if d.get("modalities"):
        fields["modalities"] = d["modalities"]
    if d.get("splits"):
        fields["split_names"] = d["splits"]
    # Easy mode always uses the manually-assigned splits so train/val/test are proper
    # (run scripts/assign_splits.py first to populate them).
    fields["split_type"] = "user_defined"

    tasks: list[str] = []
    if d.get("classification"):
        tasks.append("classification")
        fields["classification_concepts"] = d["classification"]   # DR / AMD / Glaucoma …
    if d.get("segmentation"):
        tasks.append("segmentation")
        fields["segmentation_types"] = d["segmentation"]          # od / oc / vessels / av / lesions
    if d.get("grading"):
        tasks.append("grading")
        fields["disease_types"] = d["grading"]
    if d.get("localization"):
        tasks.append("localization")
        fields["localization_types"] = d["localization"]      # bbox / keypoint / …
    if tasks:
        fields["annotation_tasks"] = tasks

    if d.get("caption") and d["caption"] != "(none)":
        fields["caption_mode"] = d["caption"]                 # vision-language captions

    rmode = d.get("require_mode")
    if rmode and rmode != "none":
        fields["require_annotations_mode"] = rmode

    hsf = d.get("health")
    if hsf and hsf != "(none)":
        fields["health_status_filter"] = hsf
        fields["include_health_status"] = True

    if d.get("good_quality_only"):
        fields["iqa_quality_labels"] = ["good", "usable"]

    return fields


def _render_flatten_code(datasets, out_path: str) -> str:
    """Render the 'flatten everything for a dataset' snippet (uses build_dataset_spec)."""
    ds_repr = _repr(datasets) if datasets else '["DATASET_NAME"]  # ← specify dataset(s)'
    return "\n".join([
        "import asyncio",
        "from chaksudb.export.discovery import build_dataset_spec",
        "from chaksudb.export.api import export",
        "",
        "# Introspects the dataset and flattens EVERYTHING it has into columns",
        "# (grading + full classification panel + per-type quality + segmentation + patient).",
        f"spec = asyncio.run(build_dataset_spec({ds_repr}))",
        f'export(spec, parquet_path="{out_path or "out.parquet"}")',
    ])


def generate_easy_code(modalities, datasets, splits, classification, segmentation, grading,
                       localization, caption, require_mode, health, good_quality_only,
                       flatten, parquet_path_str) -> str:
    out_path = (parquet_path_str or "out.parquet").strip()
    if flatten:
        return _render_flatten_code(datasets, out_path)
    fields = _assemble_easy_fields({
        "modalities": modalities, "datasets": datasets, "splits": splits,
        "classification": classification, "segmentation": segmentation,
        "grading": grading, "localization": localization, "caption": caption,
        "require_mode": require_mode,
        "health": health, "good_quality_only": good_quality_only,
    })
    return _render_spec_code(fields, out_path)


async def preview_easy_count(modalities, datasets, splits, classification, segmentation, grading,
                             localization, caption, require_mode, health, good_quality_only,
                             flatten, parquet_path_str) -> str:
    try:
        if flatten:
            from chaksudb.export.discovery import build_dataset_spec
            if not datasets:
                return "**Pick at least one dataset** to flatten."
            spec = await build_dataset_spec(datasets)
        else:
            spec = ExportSpec(**_assemble_easy_fields({
                "modalities": modalities, "datasets": datasets, "splits": splits,
                "classification": classification, "segmentation": segmentation,
                "grading": grading, "localization": localization, "caption": caption,
                "require_mode": require_mode,
                "health": health, "good_quality_only": good_quality_only,
            }))
        n = await count_rows(spec)
        return f"**{n:,} images** match your current selection."
    except Exception as e:
        return f"**Could not count images:** {e}"


# ── visibility helpers ────────────────────────────────────────────────────────

def tasks_changed(tasks):
    tasks = tasks or []
    def _section(task):
        show = task in tasks
        return gr.update(visible=show, open=True) if show else gr.update(visible=False)
    return (
        _section("grading"),
        _section("segmentation"),
        _section("classification"),
        _section("localization"),
    )

def lesion_selected(seg_types):
    return gr.update(visible="lesions" in (seg_types or []))

def iqa_score_toggled(enabled):
    return gr.update(visible=bool(enabled))


# ── build UI ─────────────────────────────────────────────────────────────────

def build_ui(db_opts: dict):
    dataset_choices    = db_opts.get("datasets", [])
    dataset_modalities = db_opts.get("dataset_modalities", {})
    class_name_choices = db_opts.get("class_names", [])
    db_error           = db_opts.get("db_error")

    if db_error:
        db_banner = (
            "> ⚠ **Could not reach the database** — dataset list is unavailable.  \n"
            "> Make sure the database is running and you have the correct connection settings, then restart this tool."
        )
    elif dataset_choices:
        db_banner = f"> ✓ Connected — **{len(dataset_choices)} datasets** available."
    else:
        db_banner = "> ⚠ Database is reachable but no datasets have been loaded yet."

    with gr.Blocks(title="ChaksuDB Export Builder") as demo:
        db_state = gr.State({
            "datasets": dataset_choices,
            "dataset_modalities": dataset_modalities,
        })

        gr.Markdown("# ChaksuDB Export Builder")
        gr.Markdown(db_banner)

        with gr.Tabs():

            # ════════════════════════════════════════════════════════════════
            # EASY TAB
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("Easy"):
                gr.Markdown(
                    "Pick a **modality**, optional **datasets** (leave empty for *all*), and the "
                    "**tasks** you want. Or tick **Flatten everything** to dump a whole dataset "
                    "as flat columns. Then **Generate Code**."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        easy_modalities = gr.CheckboxGroup(
                            MODALITIES, label="Imaging modality",
                            info="Leave empty to include every modality.",
                            elem_id="easy_modalities",
                        )
                        easy_datasets = gr.Dropdown(
                            dataset_choices, multiselect=True, label="Datasets (optional)",
                            info="Leave empty to export from ALL datasets.",
                            elem_id="easy_datasets",
                        )
                        easy_splits = gr.CheckboxGroup(
                            SPLITS, label="Splits (optional)",
                            info="Leave empty for all splits.",
                        )

                        gr.Markdown("#### Tasks — pick one or more (each is a separate task)")
                        with gr.Accordion("🏷  Disease classification", open=True):
                            easy_classification = gr.CheckboxGroup(
                                CONCEPTS, label="Diseases (present / absent)",
                                info="Unified across datasets → one {concept}_present column each.",
                                elem_id="easy_classification",
                            )
                        with gr.Accordion("🎚  Disease severity grading", open=False, elem_id="easy_grading_acc"):
                            easy_grading = gr.CheckboxGroup(
                                DISEASE_TYPES, label="Diseases to grade",
                                info="Normalized severity grade per disease.",
                                elem_id="easy_grading",
                            )
                        with gr.Accordion("🖌  Segmentation masks", open=False):
                            easy_segmentation = gr.CheckboxGroup(
                                EASY_SEG_TYPES, label="Structures to segment",
                            )
                        with gr.Accordion("📍  Object localization", open=False):
                            easy_localization = gr.CheckboxGroup(
                                LOC_TYPES, label="Location types",
                                info="Bounding boxes / keypoints / etc.",
                            )
                        with gr.Accordion("💬  Vision-language captions", open=False):
                            easy_caption = gr.Dropdown(
                                CAPTION_MODES, value="(none)", label="Caption type",
                                info="Adds a text caption column for vision-language training.",
                            )
                        easy_require = gr.Radio(
                            EASY_REQUIRE_MODES, value="any", label="Which images to include?",
                            info="'At least one' keeps images that have any of the labels you picked above. "
                                 "'ALL' keeps only images that have every one of them.",
                        )
                        easy_health = gr.Radio(
                            HEALTH_FILTERS, value="(none)", label="Normal / diseased filter",
                            info="Export only normal, only diseased, or all (adds a health_status column).",
                        )
                        easy_good_quality = gr.Checkbox(
                            False, label="Only good/usable quality images",
                        )

                        easy_flatten = gr.Checkbox(
                            False, label="⚡ Flatten EVERYTHING for the selected dataset(s)",
                            info="Ignore the task pickers and export every label the dataset has "
                                 "(uses build_dataset_spec). Needs at least one dataset.",
                        )
                        easy_path = gr.Textbox(
                            value="out.parquet", label="Output file path",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Code")
                        easy_generate_btn = gr.Button("Generate Code", variant="primary", size="lg",
                                                      elem_id="easy_generate_btn")
                        easy_code_out = gr.Code(language="python", label="", lines=24, interactive=False)
                        easy_count_btn = gr.Button("Count matching images", variant="primary",
                                                   elem_id="easy_count_btn")
                        easy_count_out = gr.Markdown("_Click to count images for your selection._")

                easy_inputs = [
                    easy_modalities, easy_datasets, easy_splits, easy_classification, easy_segmentation,
                    easy_grading, easy_localization, easy_caption, easy_require,
                    easy_health, easy_good_quality, easy_flatten, easy_path,
                ]
                easy_generate_btn.click(generate_easy_code, inputs=easy_inputs, outputs=easy_code_out)
                easy_count_btn.click(preview_easy_count, inputs=easy_inputs, outputs=easy_count_out)

            # ════════════════════════════════════════════════════════════════
            # ADVANCED TAB
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("Advanced"):
                gr.Markdown(
                    "Every ExportSpec knob. Click **Generate Code** to produce ready-to-run Python, "
                    "or **Count matching images** to preview the size."
                )
                with gr.Row(equal_height=False):

                    # ── LEFT COLUMN: controls ─────────────────────────────────
                    with gr.Column(scale=1):

                        with gr.Accordion("Imaging Modality", open=True):
                            modalities = gr.CheckboxGroup(
                                MODALITIES, label="What type of images?",
                                info="Select one or more modalities; the dataset list narrows to matching datasets. Leave empty for all.",
                            )

                        with gr.Accordion("Datasets & Splits", open=True):
                            with gr.Row():
                                dataset_names = gr.Dropdown(
                                    dataset_choices, multiselect=True, label="Which datasets?",
                                    info="Filtered by modality above. Leave empty to include all shown datasets.",
                                    scale=4,
                                )
                                with gr.Column(scale=1, min_width=120):
                                    gr.Markdown("&nbsp;")
                                    select_all_btn = gr.Button("Select all", size="sm")
                                    clear_btn = gr.Button("Clear", size="sm")
                            split_names = gr.CheckboxGroup(
                                SPLITS, label="Which splits?",
                                info="Leave empty to include all splits (train + val + test).",
                            )
                            split_type = gr.Dropdown(
                                SPLIT_TYPES, value="(any)", label="Split assignment method",
                                info="Choose 'Manually assigned' to get a proper train / val / test split.",
                            )

                        with gr.Accordion("What to Export", open=True):
                            annotation_tasks = gr.CheckboxGroup(
                                ANNOTATION_TASKS, label="Annotation types to include",
                                info="Select one or more. Each type adds extra columns to the export.",
                            )
                            require_annotations_mode = gr.Radio(
                                REQUIRE_MODES, value="none", label="Which images to include?",
                            )
                            annotation_source = gr.Dropdown(
                                ANNOTATION_SOURCES, value="prefer_consensus",
                                label="Which annotations to use?",
                                info="Most datasets have both expert and consensus labels. Choose which to use.",
                            )

                        # Health status (always available)
                        with gr.Accordion("Health Status (normal / abnormal)", open=False):
                            include_health_status = gr.Checkbox(
                                False, label="Add a health_status column (normal / abnormal / unknown)",
                                info="Derived across grading + disease classification.",
                            )
                            health_status_filter = gr.Radio(
                                HEALTH_FILTERS, value="(none)", label="Keep only…",
                                info="Filter to normal or diseased images (implies the column).",
                            )

                        # Grading (conditional)
                        grading_section = gr.Accordion("Disease Severity Grades", open=True, visible=False)
                        with grading_section:
                            disease_types = gr.CheckboxGroup(
                                DISEASE_TYPES, label="Which diseases?",
                                info="Leave empty to include all graded diseases.",
                            )
                            with gr.Row():
                                include_original_grade = gr.Checkbox(True, label="Include original label text")
                                include_scaled_grade   = gr.Checkbox(True, label="Include numeric severity score")
                            grade_filter_json = gr.Textbox(
                                label="Severity range filter  (optional)",
                                placeholder='{"DR": {"min": 1, "max": 3}}  or  {"DR": [0, 1, 2]}',
                                info='Restrict to specific severity levels.',
                                lines=2,
                            )

                        # Segmentation (conditional)
                        seg_section = gr.Accordion("Segmentation Masks", open=True, visible=False)
                        with seg_section:
                            segmentation_types = gr.CheckboxGroup(
                                SEG_TYPES, label="Which structures to segment?",
                                info="Leave empty to include all available mask types.",
                            )
                            lesion_section = gr.Group(visible=False)
                            with lesion_section:
                                lesion_subtypes = gr.CheckboxGroup(
                                    LESION_SUBTYPES, label="Which lesion types?",
                                    info="Leave empty for all lesion subtypes.",
                                )

                        # Classification (conditional)
                        cls_section = gr.Accordion("Classification Labels", open=True, visible=False)
                        with cls_section:
                            classification_concepts = gr.CheckboxGroup(
                                CONCEPTS, label="Disease concepts (recommended)",
                                info="Unified per-disease present/absent columns across all datasets.",
                            )
                            classification_positive_for = gr.CheckboxGroup(
                                CONCEPTS, label="Keep only images POSITIVE for…",
                                info="Cross-dataset positivity filter (any storage shape).",
                            )
                            cls_info = (
                                "Exact class_name pivots (advanced). Usually you want the concepts above instead."
                                if class_name_choices else
                                "No classification data found, or the database is not connected."
                            )
                            classification_class_names_raw = gr.CheckboxGroup(
                                class_name_choices, label="Exact class names (optional)", info=cls_info,
                            )
                            classification_label_type = gr.Radio(
                                LABEL_TYPES, value="int", label="Label format",
                            )

                        # Localization (conditional)
                        loc_section = gr.Accordion("Object Locations", open=True, visible=False)
                        with loc_section:
                            localization_types = gr.CheckboxGroup(
                                LOC_TYPES, label="Which location types?",
                                info="Leave empty for all available location annotations.",
                            )
                            detection_format = gr.Radio(
                                DETECTION_FORMATS, value="nested", label="Output format",
                                info="COCO format is needed for tools like Detectron2 or MMDetection.",
                            )

                        # Quality / IQA
                        with gr.Accordion("Image Quality", open=False):
                            quality_types = gr.CheckboxGroup(
                                QUALITY_TYPE_OPTS, label="Quality types to pivot into columns",
                                info="Leave empty to use the default set. Adds {type}_quality_score/_label columns.",
                            )
                            iqa_min_score_enabled = gr.Checkbox(False, label="Only include good-quality images")
                            iqa_score_row = gr.Row(visible=False)
                            with iqa_score_row:
                                iqa_min_quality_score = gr.Slider(
                                    0.0, 1.0, value=0.5, step=0.05,
                                    label="Minimum quality threshold (0 = any, 1 = perfect only)",
                                )
                            iqa_quality_labels = gr.CheckboxGroup(
                                IQA_LABELS, label="Quality tiers to include",
                                info="Leave empty to skip quality filtering.",
                            )
                            include_fundus_roi = gr.Checkbox(
                                False, label="Include fundus region-of-interest (ROI) coordinates",
                            )

                        # Output options
                        with gr.Accordion("Extra Columns & Output", open=False):
                            include_patient_data = gr.Checkbox(False, label="Include patient demographics (age, sex, ethnicity)")
                            caption_mode = gr.Dropdown(
                                CAPTION_MODES, value="(none)", label="Add text captions",
                            )
                            output_format = gr.Dropdown(
                                OUTPUT_FORMATS, value="(none)", label="PyTorch output format",
                            )
                            base_path_for_paths = gr.Textbox(
                                label="Image path prefix  (optional)",
                                placeholder="/mnt/data  or  http://192.168.1.10:8091",
                            )

                        with gr.Accordion("Save Location", open=True):
                            parquet_path_str = gr.Textbox(
                                value="out.parquet", label="Output file path",
                            )

                    # ── RIGHT COLUMN: generated code ──────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Code")
                        generate_btn = gr.Button("Generate Code", variant="primary", size="lg")
                        code_out = gr.Code(language="python", label="", lines=32, interactive=False)
                        gr.Markdown("### How many images will this export?")
                        count_btn = gr.Button("Count matching images", variant="primary")
                        count_out = gr.Markdown("_Click the button above to check how many images match._")

                # ── Advanced inputs — ORDER MUST MATCH _ADVANCED_KEYS ──────────
                advanced_inputs = [
                    dataset_names, split_names, split_type,
                    modalities,
                    annotation_tasks, require_annotations_mode, annotation_source,
                    disease_types, include_original_grade, include_scaled_grade, grade_filter_json,
                    segmentation_types, lesion_subtypes,
                    classification_class_names_raw, classification_label_type,
                    localization_types, detection_format,
                    iqa_min_score_enabled, iqa_min_quality_score,
                    iqa_quality_labels, include_fundus_roi,
                    include_patient_data, caption_mode, output_format,
                    base_path_for_paths,
                    parquet_path_str,
                    classification_concepts, classification_positive_for, quality_types,
                    include_health_status, health_status_filter,
                ]

                # modality → dataset cascade
                async def _on_modality_change(selected_modalities, current_dataset_names, state):
                    try:
                        from chaksudb.db.connection import close_pool
                        all_ds, dm = await _query_dataset_modalities()
                        await close_pool()
                        new_state = {"datasets": all_ds, "dataset_modalities": dm}
                    except Exception:
                        all_ds = state.get("datasets", [])
                        dm = state.get("dataset_modalities", {})
                        new_state = state
                    filtered = _datasets_for_modalities(selected_modalities, all_ds, dm)
                    if selected_modalities:
                        extras = [d for d in (current_dataset_names or []) if d in filtered]
                        new_value = list(dict.fromkeys(filtered + extras))
                    else:
                        new_value = [d for d in (current_dataset_names or []) if d in filtered]
                    return gr.update(choices=filtered, value=new_value), new_state

                modalities.change(
                    _on_modality_change,
                    inputs=[modalities, dataset_names, db_state],
                    outputs=[dataset_names, db_state],
                )

                def _select_all(selected_modalities, state):
                    all_ds = state.get("datasets", dataset_choices)
                    dm = state.get("dataset_modalities", dataset_modalities)
                    return gr.update(value=_datasets_for_modalities(selected_modalities, all_ds, dm))

                select_all_btn.click(_select_all, inputs=[modalities, db_state], outputs=[dataset_names])
                clear_btn.click(lambda: gr.update(value=[]), outputs=[dataset_names])

                annotation_tasks.change(
                    tasks_changed, inputs=[annotation_tasks],
                    outputs=[grading_section, seg_section, cls_section, loc_section],
                )
                segmentation_types.change(lesion_selected, inputs=[segmentation_types], outputs=[lesion_section])
                iqa_min_score_enabled.change(iqa_score_toggled, inputs=[iqa_min_score_enabled], outputs=[iqa_score_row])

                generate_btn.click(generate_code, inputs=advanced_inputs, outputs=code_out)
                count_btn.click(preview_count, inputs=advanced_inputs, outputs=count_out)

    return demo


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChaksuDB Export Builder")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  default="0.0.0.0",
                        help="Bind address (0.0.0.0 = accessible on LAN)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio tunnel URL")
    args = parser.parse_args()

    print("Connecting to DB to load option lists …")
    db_opts = fetch_db_options()
    if db_opts.get("db_error"):
        print(f"  ⚠  DB unreachable — dropdowns will be empty. ({db_opts['db_error']})")
    else:
        n_ds = len(db_opts.get("datasets", []))
        n_cls = len(db_opts.get("class_names", []))
        print(f"  ✓  {n_ds} datasets, {n_cls} classification classes loaded.")

    demo = build_ui(db_opts)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
