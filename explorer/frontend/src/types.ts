export type DatasetOption = {
  name: string
  config_fields: Array<{ name: string; type: string; required: boolean; default: unknown }>
}

export type ExplorerSettings = {
  base_dir: string | null
  resolved_base_dir: string
  default_base_dir: string
  default_cache_root: string
}

export type PreparedDataset = {
  id: string
  dataset_class_name: string
  config_name: string
  config_hash: string
  storage_type: string
  data_model: string
  created_at: string | null
  path: string
  relative_path: string
  splits: Record<string, number>
  metadata: { description: string; homepage: string; license: string }
}

export type PreparedDatasetInventory = ExplorerSettings & {
  datasets: PreparedDataset[]
  invalid_snapshots: Array<{ path: string; reason: string }>
  scanned_at: string
}

export type PreparationJob = {
  id: string
  status: string
  dataset_name: string
  source_dir: string | null
  output_root: string
  config: Record<string, unknown>
  created_at: string
  started_at: string | null
  finished_at: string | null
  result_path: string | null
  error: string | null
}

export type LoadedDataset = PreparedDataset

export type DocumentSummary = {
  index: number
  split: string
  sample_id: string
  transcription_preview: string
  annotation_counts: Record<string, number>
  warning_count: number
  image_url: string
}

export type DocumentPage = DocumentSummary & {
  dataset: string
  dataset_id: string
  data_dir: string | null
  image_path: string | null
  width: number
  height: number
  transcription: string
  warnings: string[]
  ocr_row_count: number
  ocr_rows: Array<{
    id: number
    parent_id: number | null
    level: string | number | null
    text: string
    bbox: number[] | null
  }>
}
