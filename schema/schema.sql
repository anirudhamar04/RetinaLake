-- ============================================
-- Extensions
-- ============================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================
-- UUID Strategy
-- ============================================
-- All primary keys use UUID v5 (deterministic) generated in application code.
-- UUIDs are computed from identifying fields (dataset_name, original_image_id, etc.)
-- to enable hash-based lookups and natural idempotency.
-- No DEFAULT clauses are used - UUIDs must be provided in INSERT statements.

-- ============================================
-- Core: Datasets
-- ============================================
CREATE TABLE datasets (
  dataset_id UUID PRIMARY KEY,
  dataset_name TEXT NOT NULL,
  source_url TEXT,
  license TEXT,
  modality_types TEXT[] DEFAULT '{}',
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

-- ============================================
-- Core: Models (optional labelers)
-- ============================================
CREATE TABLE models (
  model_id UUID PRIMARY KEY,
  model_name TEXT NOT NULL,
  model_description TEXT,
  model_url TEXT
);

-- ============================================
-- Core: Experts
-- ============================================s
-- updated: allow dataset_id OR model_id (pseudo expert)
CREATE TABLE experts (
  expert_id UUID PRIMARY KEY,
  expert_name TEXT,
  expertise_area TEXT,

  -- Either:
  -- dataset_id != NULL => real source expert
  -- model_id != NULL   => pseudo expert (model)
  dataset_id UUID,
  model_id UUID,

  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_experts_dataset
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),

  CONSTRAINT fk_experts_model
    FOREIGN KEY (model_id) REFERENCES models(model_id),

  CONSTRAINT ck_experts_source
    CHECK (dataset_id IS NOT NULL OR model_id IS NOT NULL)
);

CREATE INDEX idx_experts_dataset_id ON experts(dataset_id);
CREATE INDEX idx_experts_model_id ON experts(model_id);

-- ============================================
-- Patients
-- ============================================
CREATE TABLE patients (
  patient_id UUID PRIMARY KEY,
  dataset_id UUID NOT NULL,
  original_patient_id TEXT NOT NULL,

  age INTEGER,
  sex TEXT,
  ethnicity TEXT,
  nationality TEXT,
  comorbidities JSONB,

  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_patients_dataset
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),

  CONSTRAINT ck_patients_sex
    CHECK (sex IS NULL OR sex IN ('male', 'female', 'unknown'))
);

CREATE INDEX idx_patients_dataset_id ON patients(dataset_id);
CREATE INDEX idx_patients_original_id ON patients(original_patient_id);

-- ✅ prevent duplicates only when original_patient_id is meaningful
-- NOTE: original_patient_id is NOT NULL in your schema, so this is safe.
ALTER TABLE patients
ADD CONSTRAINT uq_patients_dataset_original
UNIQUE (dataset_id, original_patient_id);

-- ============================================
-- Image groups (OCT volumes / sequences)
-- ============================================
CREATE TABLE image_groups (
  group_id UUID PRIMARY KEY,
  dataset_id UUID NOT NULL REFERENCES datasets(dataset_id),

  -- oct_volume: OCT frames
  -- video: temporal sequence
  -- sequence: generic frame series
  group_type TEXT NOT NULL CHECK (group_type IN ('oct_volume','video','sequence')),

  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_image_groups_dataset ON image_groups(dataset_id);

-- ============================================
-- Images
-- ============================================
CREATE TABLE images (
  image_id UUID PRIMARY KEY,
  dataset_id UUID NOT NULL,
  original_image_id TEXT,

  -- UPDATED storage locator (instead of single file_path)
  storage_provider TEXT NOT NULL DEFAULT 'local',
  bucket TEXT,
  object_key TEXT,
  version_id TEXT,
  file_path TEXT, -- optional for local disk workflows

  file_format TEXT,
  modality TEXT,

  -- Duplicate detection hashes (cross-dataset). Three levels of strictness:
  --   file_hash    SHA-256 of the raw bytes — exact file identity (changes on re-save).
  --   content_hash SHA-256 of the DECODED RGB pixels — encoding-invariant, so the same
  --                image stored under two lossless encodings (PNG/BMP/re-save) matches.
  --                This is the canonical dedup key.
  --   phash        64-bit perceptual (dHash) hex — matches the same image even across
  --                lossy re-encoding (JPEG) or a resize; used for near-duplicate auditing
  --                (compare by Hamming distance), not auto-merge.
  file_hash TEXT,
  content_hash TEXT,
  phash TEXT,

  -- OCT volume support
  group_id UUID,
  frame_index INTEGER,

  resolution_width INTEGER,
  resolution_height INTEGER,
  field_of_view INTEGER,

  eye_laterality TEXT,
  acquisition_date DATE,

  created_at TIMESTAMP NOT NULL DEFAULT now(),
  updated_at TIMESTAMP,

  CONSTRAINT fk_images_dataset
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),

  CONSTRAINT fk_images_group
    FOREIGN KEY (group_id) REFERENCES image_groups(group_id),

  CONSTRAINT ck_images_file_format
    CHECK (
      file_format IS NULL OR
      file_format IN ('jpg','jpeg','png','tif','tiff','ppm','dicom')
    ),

  CONSTRAINT ck_images_modality
    CHECK (
      modality IS NULL OR
      modality IN ('fundus','oct','fa','uwf')
    ),

  CONSTRAINT ck_images_eye_laterality
    CHECK (
      eye_laterality IS NULL OR
      eye_laterality IN ('left','right','unknown')
    ),

  CONSTRAINT ck_images_storage_provider
    CHECK (storage_provider IN ('local','s3','gcs','azure','http')),

  -- enforce at least one locator exists
  CONSTRAINT ck_images_locator
    CHECK (
      (storage_provider = 'local' AND file_path IS NOT NULL)
      OR
      (storage_provider <> 'local' AND object_key IS NOT NULL)
    )
);

CREATE INDEX idx_images_dataset_id ON images(dataset_id);
CREATE INDEX idx_images_group_frame ON images(group_id, frame_index);
CREATE INDEX idx_images_original_image_id ON images(original_image_id);


-- ✅ uniqueness without breaking NULL original_image_id ingestion
CREATE UNIQUE INDEX uq_images_dataset_original_image_id
ON images(dataset_id, original_image_id)
WHERE original_image_id IS NOT NULL;

-- ============================================
-- Cross-dataset image membership
-- The same physical file may appear in multiple datasets. The first dataset to ingest
-- it owns the canonical images row; subsequent datasets record a membership here
-- (pointing at the canonical image_id) instead of inserting a duplicate images row.
-- ============================================
CREATE TABLE image_dataset_memberships (
  image_id UUID NOT NULL,
  dataset_id UUID NOT NULL,
  original_image_id TEXT,            -- id used by the secondary dataset
  added_at TIMESTAMP NOT NULL DEFAULT now(),

  PRIMARY KEY (image_id, dataset_id),

  CONSTRAINT fk_idm_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),
  CONSTRAINT fk_idm_dataset
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

CREATE INDEX idx_idm_dataset ON image_dataset_memberships(dataset_id);

-- ============================================
-- Patient <-> Image relationship (join table)
-- ============================================
CREATE TABLE patient_images (
  relationship_id UUID PRIMARY KEY,
  patient_id UUID NOT NULL,
  image_id UUID NOT NULL,
  exam_date DATE,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_patient_images_patient
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),

  CONSTRAINT fk_patient_images_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT uq_patient_images_unique_pair
    UNIQUE (patient_id, image_id)
);

CREATE INDEX idx_patient_images_patient ON patient_images(patient_id);
CREATE INDEX idx_patient_images_image ON patient_images(image_id);

-- ============================================
-- Raw annotation files
-- ============================================
CREATE TABLE raw_annotation_files (
  raw_file_id UUID PRIMARY KEY,
  dataset_id UUID NOT NULL,

  -- UPDATED storage locator
  storage_provider TEXT NOT NULL DEFAULT 'local',
  bucket TEXT,
  object_key TEXT,
  version_id TEXT,
  file_path TEXT,

  file_type TEXT,
  file_name TEXT,
  file_hash TEXT,
  file_size BIGINT,

  encoding TEXT,
  parsed_status TEXT NOT NULL DEFAULT 'not_parsed',
  parse_errors TEXT,

  created_at TIMESTAMP NOT NULL DEFAULT now(),
  updated_at TIMESTAMP,

  CONSTRAINT fk_raw_files_dataset
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),

  CONSTRAINT ck_raw_files_type
    CHECK (
      file_type IS NULL OR
      file_type IN ('csv','json','xml','excel','txt','mat','html','jsonl','xlsx','xls','wmv','jpg','jpeg','png','tif','tiff','ppm','dicom','JPG','JPEG','PNG','TIF','TIFF','PPM','DICOM')
    ),

  CONSTRAINT ck_raw_files_parsed_status
    CHECK (
      parsed_status IN ('not_parsed','parsed','error')
    ),

  CONSTRAINT ck_raw_files_storage_provider
    CHECK (storage_provider IN ('local','s3','gcs','azure','http')),

  CONSTRAINT ck_raw_files_locator
    CHECK (
      (storage_provider = 'local' AND file_path IS NOT NULL)
      OR
      (storage_provider <> 'local' AND object_key IS NOT NULL)
    )
);

CREATE INDEX idx_raw_files_dataset ON raw_annotation_files(dataset_id);
CREATE INDEX idx_raw_files_hash ON raw_annotation_files(file_hash);

-- ✅ prevent duplicate ingestion of same file per dataset
CREATE UNIQUE INDEX uq_raw_files_dataset_hash
ON raw_annotation_files(dataset_id, file_hash)
WHERE file_hash IS NOT NULL;

-- ============================================
-- Expert annotations (metadata)
-- Parent table: task tables will reference this.
-- ============================================
CREATE TABLE expert_annotations (
  expert_annotation_id UUID PRIMARY KEY,
  expert_id UUID NOT NULL,

  annotation_task TEXT NOT NULL,
  raw_data_id UUID,

  annotation_value JSONB,

  confidence_level TEXT,
  annotation_timestamp TIMESTAMP,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_expert_annotations_expert
    FOREIGN KEY (expert_id) REFERENCES experts(expert_id),

  CONSTRAINT fk_expert_annotations_raw
    FOREIGN KEY (raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT ck_expert_annotations_task
    CHECK (annotation_task IN ('grading','segmentation','classification','localization','quality','keyword','description')),

  CONSTRAINT ck_expert_annotations_confidence
    CHECK (confidence_level IS NULL OR confidence_level IN ('high','medium','low'))
);

CREATE INDEX idx_expert_annotations_expert ON expert_annotations(expert_id);
CREATE INDEX idx_expert_annotations_raw ON expert_annotations(raw_data_id);

-- ============================================
-- Annotation Type Vocabulary (for segmentation types)
-- ============================================
CREATE TABLE annotation_type (
  annotation_type_id UUID PRIMARY KEY,
  annotation_type TEXT NOT NULL,
  annotation_description TEXT
);

-- ============================================
-- Grading scales
-- ============================================
CREATE TABLE grading_scales (
  scale_id UUID PRIMARY KEY,
  scale_name TEXT NOT NULL,
  disease_type TEXT NOT NULL,
  scale_description TEXT,
  min_value INTEGER,
  max_value INTEGER,
  value_labels JSONB,

  CONSTRAINT ck_grading_scales_disease
    CHECK (disease_type IN ('DR','DME','Glaucoma','AMD','myopic_maculopathy'))
);

-- ============================================
-- Grading scale mappings
-- ============================================
CREATE TABLE grading_scale_mappings (
  mapping_id UUID PRIMARY KEY,
  source_scale_id UUID NOT NULL,
  target_scale_id UUID NOT NULL,
  source_value TEXT NOT NULL,
  target_value INTEGER,
  mapping_confidence TEXT NOT NULL DEFAULT 'exact',

  CONSTRAINT fk_gsm_source_scale
    FOREIGN KEY (source_scale_id) REFERENCES grading_scales(scale_id),

  CONSTRAINT fk_gsm_target_scale
    FOREIGN KEY (target_scale_id) REFERENCES grading_scales(scale_id),

  CONSTRAINT ck_gsm_mapping_confidence
    CHECK (mapping_confidence IN ('exact','approximate','manual_review_required'))
);

-- ============================================
-- Consensus annotations
-- ============================================
CREATE TABLE consensus_annotations (
  consensus_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,

  annotation_task TEXT NOT NULL,
  consensus_method TEXT NOT NULL,

  expert_annotation_ids UUID[],

  consensus_value JSONB,
  agreement_score FLOAT,
  disagreement_details JSONB,

  adjudicator_id UUID,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_consensus_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_consensus_adjudicator
    FOREIGN KEY (adjudicator_id) REFERENCES experts(expert_id),

  CONSTRAINT ck_consensus_task
    CHECK (annotation_task IN ('grading','segmentation','classification','localization','quality')),

  CONSTRAINT ck_consensus_method
    CHECK (consensus_method IN ('majority_vote','mean','median','staple','adjudicated','senior_review'))
);

CREATE INDEX idx_consensus_image ON consensus_annotations(image_id);

-- ============================================
-- Provenance chain
-- ============================================
CREATE TABLE provenance_chain (
  chain_id UUID PRIMARY KEY,

  unified_annotation_type TEXT NOT NULL,
  source_type TEXT NOT NULL,

  root_source_raw_data_id UUID,
  source_annotation_ids UUID[],

  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_prov_root_raw
    FOREIGN KEY (root_source_raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT ck_prov_unified_type
    CHECK (unified_annotation_type IN ('grading','segmentation','classification','localization','quality','keyword','description')),

  CONSTRAINT ck_prov_source_type
    CHECK (source_type IN ('original','transformed','pseudo_generated','consensus'))
);

-- ============================================
-- Transformation operations
-- ============================================
CREATE TABLE transformation_operations (
  transformation_id UUID PRIMARY KEY,
  operation_type TEXT NOT NULL,

  input_data JSONB,
  output_data JSONB,
  operation_parameters JSONB,

  operation_timestamp TIMESTAMP NOT NULL DEFAULT now(),
  operator TEXT,
  notes TEXT
);

-- Join table for provenance <-> transformations
CREATE TABLE provenance_transformations (
  id UUID PRIMARY KEY,
  chain_id UUID NOT NULL,
  transformation_id UUID NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_prov_trans_chain
    FOREIGN KEY (chain_id) REFERENCES provenance_chain(chain_id),

  CONSTRAINT fk_prov_trans_transformation
    FOREIGN KEY (transformation_id) REFERENCES transformation_operations(transformation_id),

  CONSTRAINT uq_prov_trans_unique
    UNIQUE (chain_id, transformation_id)
);

-- ============================================
-- Segmentation annotations
-- ============================================
CREATE TABLE segmentation_annotations (
  segmentation_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,

  annotation_type_id UUID NOT NULL,
  lesion_subtype TEXT,

  mask_file_path TEXT,
  group_id UUID,

  unified_format TEXT,
  original_format TEXT,
  original_file_path TEXT,

  raw_data_id UUID,
  coordinate_system TEXT,

  expert_annotation_id UUID,
  consensus_id UUID,
  annotation_method TEXT NOT NULL DEFAULT 'manual',

  confidence_score FLOAT,
  provenance_chain_id UUID,

  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_seg_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_seg_annotation_type
    FOREIGN KEY (annotation_type_id) REFERENCES annotation_type(annotation_type_id),

  CONSTRAINT fk_seg_raw
    FOREIGN KEY (raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT fk_seg_expert_annotation
    FOREIGN KEY (expert_annotation_id) REFERENCES expert_annotations(expert_annotation_id),

  CONSTRAINT fk_seg_consensus
    FOREIGN KEY (consensus_id) REFERENCES consensus_annotations(consensus_id),

  CONSTRAINT fk_seg_provenance
    FOREIGN KEY (provenance_chain_id) REFERENCES provenance_chain(chain_id),

  CONSTRAINT ck_seg_method
    CHECK (annotation_method IN ('manual','semi_automatic','automatic','pseudo'))
);

CREATE INDEX idx_seg_image ON segmentation_annotations(image_id);

-- ============================================
-- Disease grading annotations
-- ============================================
CREATE TABLE disease_grading (
  grading_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,

  disease_type TEXT NOT NULL,
  scale_id UUID NOT NULL,

  original_grade TEXT,
  scaled_grade INTEGER,
  grade_label TEXT,

  raw_data_id UUID,
  expert_annotation_id UUID,
  consensus_id UUID,

  annotation_method TEXT NOT NULL DEFAULT 'manual',
  confidence_score FLOAT,

  provenance_chain_id UUID,

  created_at TIMESTAMP NOT NULL DEFAULT now(),
  updated_at TIMESTAMP,

  CONSTRAINT fk_grading_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_grading_scale
    FOREIGN KEY (scale_id) REFERENCES grading_scales(scale_id),

  CONSTRAINT fk_grading_raw
    FOREIGN KEY (raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT fk_grading_expert_annotation
    FOREIGN KEY (expert_annotation_id) REFERENCES expert_annotations(expert_annotation_id),

  CONSTRAINT fk_grading_consensus
    FOREIGN KEY (consensus_id) REFERENCES consensus_annotations(consensus_id),

  CONSTRAINT fk_grading_provenance
    FOREIGN KEY (provenance_chain_id) REFERENCES provenance_chain(chain_id),

  CONSTRAINT ck_grading_disease
    CHECK (disease_type IN ('DR','DME','Glaucoma','AMD','myopic_maculopathy')),

  CONSTRAINT ck_grading_method
    CHECK (annotation_method IN ('manual','adjudicated','consensus','pseudo'))
);

CREATE INDEX idx_grading_image ON disease_grading(image_id);

-- ============================================
-- Localization annotations
-- ============================================
CREATE TABLE localization_annotations (
  localization_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,

  localization_type TEXT NOT NULL,
  target_structure TEXT NOT NULL,
  coordinates JSONB NOT NULL,

  lesion_subtype TEXT,

  raw_data_id UUID,
  expert_annotation_id UUID,
  consensus_id UUID,

  annotation_method TEXT NOT NULL DEFAULT 'manual',
  provenance_chain_id UUID,

  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_loc_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_loc_raw
    FOREIGN KEY (raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT fk_loc_expert_annotation
    FOREIGN KEY (expert_annotation_id) REFERENCES expert_annotations(expert_annotation_id),

  CONSTRAINT fk_loc_consensus
    FOREIGN KEY (consensus_id) REFERENCES consensus_annotations(consensus_id),

  CONSTRAINT fk_loc_provenance
    FOREIGN KEY (provenance_chain_id) REFERENCES provenance_chain(chain_id),

  CONSTRAINT ck_loc_type
    CHECK (localization_type IN ('bounding_box','keypoint','center_point')),

  CONSTRAINT ck_loc_method
    CHECK (annotation_method IN ('manual','pseudo'))
);

CREATE INDEX idx_loc_image ON localization_annotations(image_id);

-- ============================================
-- Classification class registry
-- ============================================
CREATE TABLE classification_classes (
  class_name TEXT PRIMARY KEY,
  task_type TEXT NOT NULL,
  display_name TEXT,
  description TEXT,
  label_set JSONB,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT ck_cc_task_type
    CHECK (task_type IN ('binary','multi_class','multi_label'))
);

-- ============================================
-- Classification annotations
-- ============================================
CREATE TABLE classification_annotations (
  classification_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,

  task_type TEXT NOT NULL,
  -- Task identity: the dataset's task this row belongs to (e.g. 'glaucoma_screening',
  -- 'fives_disease'). Drives task-aware export columns. Distinct from class_name, which
  -- is the label/class within the task.
  task_name TEXT NOT NULL,
  class_name TEXT NOT NULL,

  -- Canonical concept for cross-task filtering (e.g. 'glaucoma' for both binary AIROGS
  -- and multi_label PAPILA). NULL when the row maps to no shared concept.
  concept TEXT,

  -- Explicit shape flag (redundant with task_type but kept for transparency at query time)
  is_multilabel BOOLEAN NOT NULL,

  -- Scalar label columns (uniform across all task types; always populated)
  class_index INT NOT NULL,
  class_label TEXT NOT NULL,

  -- Multi-label sub-key (NULL for binary/multi_class)
  sub_key TEXT,

  -- Original JSONB for raw data, soft labels, or backward compatibility
  class_value JSONB,

  raw_data_id UUID,
  expert_annotation_id UUID,
  consensus_id UUID,

  annotation_method TEXT NOT NULL DEFAULT 'manual',
  confidence_score FLOAT,

  provenance_chain_id UUID,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_cls_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_cls_raw
    FOREIGN KEY (raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT fk_cls_expert_annotation
    FOREIGN KEY (expert_annotation_id) REFERENCES expert_annotations(expert_annotation_id),

  CONSTRAINT fk_cls_consensus
    FOREIGN KEY (consensus_id) REFERENCES consensus_annotations(consensus_id),

  CONSTRAINT fk_cls_provenance
    FOREIGN KEY (provenance_chain_id) REFERENCES provenance_chain(chain_id),

  CONSTRAINT ck_cls_task_type
    CHECK (task_type IN ('binary','multi_class','multi_label')),

  CONSTRAINT ck_cls_method
    CHECK (annotation_method IN ('manual','adjudicated','consensus','pseudo'))
);

CREATE INDEX idx_cls_image ON classification_annotations(image_id);
CREATE INDEX idx_cls_class_name ON classification_annotations(class_name);
CREATE INDEX idx_cls_task_class ON classification_annotations(task_type, class_name);
CREATE INDEX idx_cls_subkey ON classification_annotations(class_name, sub_key) WHERE sub_key IS NOT NULL;
CREATE INDEX idx_cls_task_name ON classification_annotations(task_name);
CREATE INDEX idx_cls_concept ON classification_annotations(concept) WHERE concept IS NOT NULL;

-- ============================================
-- Quality type vocabulary (reference table)
-- Extensible registry of quality_type values (replaces a hard-coded CHECK). New types
-- are added by inserting a row — no schema migration — and the export discovers which
-- types actually exist instead of emitting a column for every possible one.
-- ============================================
CREATE TABLE quality_types (
  quality_type TEXT PRIMARY KEY,
  description TEXT,
  category TEXT,                       -- 'overall' | 'acquisition' | 'anatomical' | 'pseudo'
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

INSERT INTO quality_types (quality_type, description, category) VALUES
  ('overall',          'Overall image quality',                 'overall'),
  ('gradability',      'Whether the image is gradable',         'overall'),
  ('clarity',          'Focus / sharpness',                     'acquisition'),
  ('field_definition', 'Field-of-view adequacy',                'acquisition'),
  ('artifact',         'Presence of artifacts',                 'acquisition'),
  ('contrast',         'Image contrast',                        'acquisition'),
  ('blur',             'Blur',                                  'acquisition'),
  ('illumination',     'Illumination adequacy',                 'acquisition'),
  ('pseudo_quality',   'Model-predicted (QuickQual) quality',   'pseudo')
ON CONFLICT (quality_type) DO NOTHING;

-- ============================================
-- Quality annotations
-- ============================================
CREATE TABLE quality_annotations (
  quality_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,

  quality_type TEXT NOT NULL,
  quality_score FLOAT,
  quality_label TEXT,
  scale_description TEXT,

  raw_data_id UUID,
  expert_annotation_id UUID,

  provenance_chain_id UUID,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_quality_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_quality_raw
    FOREIGN KEY (raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT fk_quality_expert_annotation
    FOREIGN KEY (expert_annotation_id) REFERENCES expert_annotations(expert_annotation_id),

  CONSTRAINT fk_quality_provenance
    FOREIGN KEY (provenance_chain_id) REFERENCES provenance_chain(chain_id),

  -- quality_type is validated against the reference table (not a hard-coded CHECK)
  CONSTRAINT fk_quality_type
    FOREIGN KEY (quality_type) REFERENCES quality_types(quality_type)
);

CREATE INDEX idx_quality_image ON quality_annotations(image_id);
CREATE INDEX idx_quality_type ON quality_annotations(quality_type);

-- ============================================
-- Clinical descriptions
-- ============================================
CREATE TABLE clinical_descriptions (
  description_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,

  description_text TEXT NOT NULL,
  description_type TEXT NOT NULL,

  raw_data_id UUID,
  expert_id UUID,

  word_count INTEGER,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_desc_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_desc_raw
    FOREIGN KEY (raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT fk_desc_expert
    FOREIGN KEY (expert_id) REFERENCES experts(expert_id),

  CONSTRAINT ck_desc_type
    CHECK (description_type IN ('clinical_caption','diagnosis_text','notes'))
);

CREATE INDEX idx_desc_image ON clinical_descriptions(image_id);

-- ============================================
-- Keywords
-- ============================================
CREATE TABLE keyword_vocabulary (
  keyword_id UUID PRIMARY KEY,
  keyword_term TEXT NOT NULL,  -- Remove UNIQUE here
  keyword_source TEXT NOT NULL,
  category TEXT,
  dataset_id UUID NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_keyword_vocab_dataset
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),

  CONSTRAINT ck_keyword_source
    CHECK (keyword_source IN ('diagnostic_keywords','clinical_description','diagnosis_text')),
  
  CONSTRAINT uq_keyword_vocab_term_source_dataset 
    UNIQUE (keyword_term, keyword_source, dataset_id)
);

CREATE INDEX idx_keyword_vocab_dataset ON keyword_vocabulary(dataset_id);

CREATE TABLE keyword_annotations (
  keyword_annotation_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,
  keyword_id UUID NOT NULL,

  keyword_text TEXT,
  raw_data_id UUID,
  expert_id UUID,

  annotation_method TEXT NOT NULL DEFAULT 'manual',
  provenance_chain_id UUID,

  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_keyword_ann_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_keyword_ann_keyword
    FOREIGN KEY (keyword_id) REFERENCES keyword_vocabulary(keyword_id),

  CONSTRAINT fk_keyword_ann_raw
    FOREIGN KEY (raw_data_id) REFERENCES raw_annotation_files(raw_file_id),

  CONSTRAINT fk_keyword_ann_expert
    FOREIGN KEY (expert_id) REFERENCES experts(expert_id),

  CONSTRAINT fk_keyword_ann_prov
    FOREIGN KEY (provenance_chain_id) REFERENCES provenance_chain(chain_id),

  CONSTRAINT ck_keyword_ann_method
    CHECK (annotation_method IN ('manual','extracted','pseudo'))
);

CREATE INDEX idx_keyword_ann_created_at ON keyword_annotations(created_at);
CREATE INDEX idx_keyword_ann_image ON keyword_annotations(image_id);

-- ============================================
-- Dataset splits
-- ============================================
CREATE TABLE dataset_splits (
  split_id UUID PRIMARY KEY,
  dataset_id UUID NOT NULL,

  split_name TEXT NOT NULL,
  split_type TEXT NOT NULL,

  task_type TEXT,
  image_count INTEGER,

  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_dataset_splits_dataset
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),

  CONSTRAINT ck_split_type
    CHECK (split_type IN ('explicit','metadata_defined','user_defined','undefined'))
);

CREATE INDEX idx_dataset_splits_dataset ON dataset_splits(dataset_id);

CREATE TABLE image_splits (
  assignment_id UUID PRIMARY KEY,
  image_id UUID NOT NULL,
  split_id UUID NOT NULL,

  task_type TEXT,
  is_primary BOOLEAN NOT NULL DEFAULT true,

  created_at TIMESTAMP NOT NULL DEFAULT now(),

  CONSTRAINT fk_image_splits_image
    FOREIGN KEY (image_id) REFERENCES images(image_id),

  CONSTRAINT fk_image_splits_split
    FOREIGN KEY (split_id) REFERENCES dataset_splits(split_id),

  CONSTRAINT uq_image_split_unique
    UNIQUE (image_id, split_id, task_type)
);

CREATE INDEX idx_image_splits_split ON image_splits(split_id);
CREATE INDEX idx_image_splits_image ON image_splits(image_id);

-- ============================================
-- ============================================
-- TRIGGERS
-- ============================================
-- These were previously kept in schema/triggers/*.sql and applied separately
-- after schema.sql. They are folded in here so a single `psql -f schema.sql`
-- (and tests/conftest.py, which applies only schema.sql) gets the grade-conversion
-- behavior. They are defined last because they depend on the tables above
-- (disease_grading, grading_scales, grading_scale_mappings).
--
-- Trigger firing order on disease_grading (BEFORE ROW triggers fire alphabetically
-- by trigger name): set_label_from_original → trigger_auto_convert_disease_grade.
-- The two touch independent columns (grade_label vs scaled_grade), so order is safe.
-- ============================================

-- ============================================
-- Trigger: Set grade_label from original_grade via the scale's value_labels map
-- ============================================
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

COMMENT ON FUNCTION auto_convert_disease_grade() IS
'Automatically converts disease grades to target scale (ICDR_0_4) using grading_scale_mappings table.
If scale is already target scale, sets scaled_grade = original_grade.
If no mapping exists, leaves scaled_grade as NULL.';

COMMENT ON TRIGGER trigger_auto_convert_disease_grade ON disease_grading IS
'Automatically converts grades to standard scale on INSERT/UPDATE';

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

COMMENT ON FUNCTION backfill_grades_on_new_mapping() IS
'Automatically backfills disease_grading.scaled_grade when a new mapping is added to grading_scale_mappings.
Only updates records where scaled_grade IS NULL to preserve manual overrides.
Enables incremental learning of new grading scales.';

COMMENT ON TRIGGER trigger_backfill_grades_on_new_mapping ON grading_scale_mappings IS
'Backfills historical disease grades when new scale mappings are learned';
