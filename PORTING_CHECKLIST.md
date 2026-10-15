# Dataset porting checklist

Port datasets from `docdjinn-base-datasets` into the current `atria_datasets`
package, using the current `atria_core.datasets.Dataset` and
`SinglePageDocumentInstance`/annotation APIs. Port and verify one dataset at a
time before starting the next. The port scope is limited to datasets named by
`base_dataset_name` in DocDjinn's `data/syn_dataset_definitions` files.

## Semantic entity recognition (SER)

- [x] FUNSD — verified with `usage/prepare_dataset.py funsd` (149 train, 50 test)

  Run from the `atria_datasets` repository root:

  ```bash
  uv run python usage/prepare_dataset.py funsd
  ```

  This uses the repository's standard preparation and visualization flow. The
  first run downloads the original archive into Atria's default cache if it is
  not already present.
- [x] CORD — downloaded and verified with `usage/prepare_dataset.py cord`

  Run from the `atria_datasets` repository root:

  ```bash
  uv run python usage/prepare_dataset.py cord
  ```

  This downloads the original CORD parquet shards into Atria's default cache
  if they are not already present, then prepares and visualizes each split.
- [x] SROIE — verified with `usage/prepare_dataset.py sroie`

  Run from the `atria_datasets` repository root:

  ```bash
  uv run python usage/prepare_dataset.py sroie
  ```

  The first run downloads the original SROIE archive to Atria's default cache
  if it is not already present.
- [x] WildReceipts — verified with `usage/prepare_dataset.py wild_receipts`

  Run from the `atria_datasets` repository root:

  ```bash
  uv run python usage/prepare_dataset.py wild_receipts
  ```

  The first run downloads the original WildReceipt archive to Atria's default
  cache if it is not already present.

  This port was completed before the scope was narrowed to DocDjinn synth base
  datasets.
- DocILE is omitted because it is not a base dataset in the current DocDjinn
  synthetic dataset definitions.

## Other DocDjinn synth base datasets

- [x] Tobacco3482 — verified with `usage/prepare_dataset.py tobacco3482`; in use
  by `tobacco3482_alpha=1.0`

  Run from the `atria_datasets` repository root:

  ```bash
  uv run python usage/prepare_dataset.py tobacco3482
  ```

  This follows the original `image_with_ocr` configuration and downloads the
  image archive, OCR archive, and supplied train/test split files.
- [ ] RVL-CDIP — in use by `rvlcdip` configurations (large dataset)
- [ ] DUE benchmark configurations: ExDocVQA, ExKleisterCharity, and
  ExWikiTableQuestions (`ex_docvqa`, `ex_klc`, and `ex_wiki`)
- [ ] Layout analysis: PubLayNet, DocLayNet, ICDAR 2019
