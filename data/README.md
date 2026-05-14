# Data Directory

The baseline targets the `msmarco-passage/trec-dl-2019/judged` split from `ir_datasets`, with `msmarco-passage/trec-dl-2020/judged` available as a second evaluation set.

The raw MS MARCO passage collection is large, so generated dataset files are not checked in. Build the local judged-pool CSV files for both TREC-DL judged splits with:

```bash
python -m search_ranker.msmarco \
  --all-trec-dl-judged \
  --out data \
  --purge-msmarco-cache \
  --confirm-purge-msmarco-cache
```

This creates:

- `data/msmarco_trec_dl_2019_judged/corpus.csv`
- `data/msmarco_trec_dl_2019_judged/queries.csv`
- `data/msmarco_trec_dl_2019_judged/qrels.csv`
- `data/msmarco_trec_dl_2019_judged/metadata.json`
- `data/msmarco_trec_dl_2020_judged/corpus.csv`
- `data/msmarco_trec_dl_2020_judged/queries.csv`
- `data/msmarco_trec_dl_2020_judged/qrels.csv`
- `data/msmarco_trec_dl_2020_judged/metadata.json`
- `data/msmarco_trec_dl_manifest.json`

The generated corpora contain judged passages for the selected queries, not the full 8.8M passage collection. After the CSV files are written, `--purge-msmarco-cache --confirm-purge-msmarco-cache` removes only the MS MARCO `ir_datasets` cache so the large raw download does not remain on disk.
