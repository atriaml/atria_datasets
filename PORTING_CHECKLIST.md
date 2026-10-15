# Dataset porting checklist

Port datasets from `docdjinn-base-datasets` into the current `atria_datasets`
package, using the current `atria_core.datasets.Dataset` and
`SinglePageDocumentInstance`/annotation APIs. Port and verify one dataset at a
time before starting the next.

## Semantic entity recognition (SER)

- [x] FUNSD — verified with `usage/prepare_dataset.py funsd` (149 train, 50 test)

  Run from the `atria_datasets` repository root:

  ```bash
  uv run python usage/prepare_dataset.py funsd
  ```

  This uses the repository's standard preparation and visualization flow. The
  first run downloads the original archive into Atria's default cache if it is
  not already present.
- [ ] CORD
- [ ] SROIE
- [ ] WildReceipts
- [ ] DocILE

## Other dataset groups (later)

- [ ] Document classification: RVL-CDIP, Tobacco3482
- [ ] Question answering: DUE benchmark configurations
- [ ] Layout analysis: PubLayNet, DocLayNet, ICDAR 2019
- [ ] Table extraction: PubTables1M, FinTabNet, ICDAR 2013
