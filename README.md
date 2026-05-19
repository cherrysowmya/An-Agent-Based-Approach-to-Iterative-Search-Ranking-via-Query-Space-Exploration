# Agent-Based Search Ranking: BM25 Baseline

This project implements the first experiment from the proposal: a repeatable BM25 retrieval baseline for comparing classical search against later LLM reranking and agent-based iterative refinement.

## What Is Included

- A dependency-free BM25 implementation.
- CSV loaders for corpora, queries, and relevance labels.
- Ranking metrics: Precision@k, MRR@k, and nDCG@k.
- A command-line baseline runner.
- A BM25 + local Ollama reranking runner using `llama3.1:8b`.
- A BM25 + agent-based iterative refinement runner using `llama3.1:8b`.
- An MS MARCO/TREC-DL dataset converter for the 2019 and 2020 judged splits.

## Prepare MS MARCO Data

The baseline targets `msmarco-passage/trec-dl-2019/judged` from `ir_datasets`, with `msmarco-passage/trec-dl-2020/judged` available as a second evaluation set. Install the dataset helper first:

```bash
python -m pip install ".[datasets]"
```

Then materialize both local judged-document pools:

```bash
python -m search_ranker.msmarco \
  --all-trec-dl-judged \
  --out data \
  --purge-msmarco-cache \
  --confirm-purge-msmarco-cache
```

This creates `corpus.csv`, `queries.csv`, `qrels.csv`, and `metadata.json` under both `data/msmarco_trec_dl_2019_judged` and `data/msmarco_trec_dl_2020_judged`.

The generated corpora use the judged passages for the TREC-DL queries, not the full 8.8M-passage MS MARCO collection. This keeps the current in-memory BM25 baseline practical while still using the benchmark's real queries and graded qrels. The cache-purge flags remove only the raw MS MARCO `ir_datasets` cache after the smaller CSV exports are written.

## Run the Baseline

After preparing the data:

```bash
python -m search_ranker.baseline \
  --corpus data/msmarco_trec_dl_2019_judged/corpus.csv \
  --queries data/msmarco_trec_dl_2019_judged/queries.csv \
  --qrels data/msmarco_trec_dl_2019_judged/qrels.csv \
  --top-k 10 \
  --relevance-threshold 2 \
  --out outputs/msmarco_trec_dl_2019_baseline
```

The command writes:

- `outputs/msmarco_trec_dl_2019_baseline/rankings.csv`
- `outputs/msmarco_trec_dl_2019_baseline/metrics.json`
- `outputs/msmarco_trec_dl_2019_baseline/run_log.txt`

It also prints the aggregate metrics to the terminal.

## Run BM25 + LLM Reranking

Install and start Ollama, then pull the unpaid local model:

```bash
ollama pull llama3.1:8b
ollama serve
```

In another terminal, run the reranking experiment:

```bash
python -m search_ranker.rerank \
  --corpus data/msmarco_trec_dl_2019_judged/corpus.csv \
  --queries data/msmarco_trec_dl_2019_judged/queries.csv \
  --qrels data/msmarco_trec_dl_2019_judged/qrels.csv \
  --bm25-top-k 20 \
  --final-top-k 10 \
  --relevance-threshold 2 \
  --model llama3.1:8b \
  --out outputs/msmarco_trec_dl_2019_llm_rerank
```

For a quick local check before the full run, limit the query count:

```bash
python -m search_ranker.rerank \
  --corpus data/msmarco_trec_dl_2019_judged/corpus.csv \
  --queries data/msmarco_trec_dl_2019_judged/queries.csv \
  --qrels data/msmarco_trec_dl_2019_judged/qrels.csv \
  --bm25-top-k 5 \
  --final-top-k 5 \
  --relevance-threshold 2 \
  --model llama3.1:8b \
  --max-queries 3 \
  --out outputs/msmarco_trec_dl_2019_llm_rerank_quick
```

The reranker writes:

- `outputs/msmarco_trec_dl_2019_llm_rerank/reranked_rankings.csv`
- `outputs/msmarco_trec_dl_2019_llm_rerank/llm_candidate_scores.csv`
- `outputs/msmarco_trec_dl_2019_llm_rerank/metrics.json`
- `outputs/msmarco_trec_dl_2019_llm_rerank/run_log.txt`
- `outputs/msmarco_trec_dl_2019_llm_rerank/llm_score_cache.json`

The cache prevents repeated Ollama calls for the same model/query/passage combination.

## Run BM25 + Agent-Based Iterative Refinement

The agent experiment repeatedly retrieves, scores candidates, decides whether to refine the query, and stores an iteration log:

```bash
python -m search_ranker.agent \
  --corpus data/msmarco_trec_dl_2019_judged/corpus.csv \
  --queries data/msmarco_trec_dl_2019_judged/queries.csv \
  --qrels data/msmarco_trec_dl_2019_judged/qrels.csv \
  --bm25-top-k 10 \
  --final-top-k 10 \
  --max-iterations 3 \
  --score-threshold 6 \
  --improvement-epsilon 0.25 \
  --relevance-threshold 2 \
  --model llama3.1:8b \
  --max-queries 3 \
  --out outputs/msmarco_trec_dl_2019_agent_quick
```

Remove `--max-queries 3` for the full run. The agent writes:

- `agent_final_rankings.csv`
- `agent_iteration_log.csv`
- `agent_candidate_scores.csv`
- `metrics.json`
- `run_log.txt`
- `agent_score_cache.json`

## Run All Four Experiments

The suite runner executes the project report's four experiments and writes combined summaries:

```bash
python -m search_ranker.suite \
  --corpus data/msmarco_trec_dl_2019_judged/corpus.csv \
  --queries data/msmarco_trec_dl_2019_judged/queries.csv \
  --qrels data/msmarco_trec_dl_2019_judged/qrels.csv \
  --bm25-top-k 10 \
  --final-top-k 10 \
  --max-iterations 3 \
  --score-threshold 6 \
  --improvement-epsilon 0.25 \
  --relevance-threshold 2 \
  --model llama3.1:8b \
  --max-queries 3 \
  --out outputs/msmarco_trec_dl_2019_experiment_suite_quick
```

It writes:

- `experiment_1_baseline_vs_reranking.csv`
- `experiment_2_method_comparison.csv`
- `experiment_3_iteration_analysis.csv`
- `experiment_4_query_refinement_impact.csv`
- `suite_summary.json`
- subfolders containing the underlying BM25, LLM reranking, and agent runs

## Compare BM25 vs BM25 + LLM Reranking

Use one command to run both systems on the same query set and write side-by-side deltas:

```bash
python -m search_ranker.compare \
  --corpus data/msmarco_trec_dl_2019_judged/corpus.csv \
  --queries data/msmarco_trec_dl_2019_judged/queries.csv \
  --qrels data/msmarco_trec_dl_2019_judged/qrels.csv \
  --bm25-top-k 20 \
  --final-top-k 10 \
  --relevance-threshold 2 \
  --model llama3.1:8b \
  --max-queries 3 \
  --out outputs/msmarco_trec_dl_2019_comparison_quick
```

Remove `--max-queries 3` for the full comparison. The comparison output includes:

- `aggregate_comparison.csv`
- `per_query_comparison.csv`
- `comparison.json`
- `bm25/`
- `bm25_llm_rerank/`

## Dataset Format

The MS MARCO converter writes the same CSV schema used by the baseline:

Corpus CSV:

```csv
doc_id,title,text
D1,Document title,Document body text
```

Queries CSV:

```csv
query_id,query
Q1,example search query
```

Relevance labels CSV:

```csv
query_id,doc_id,relevance
Q1,D1,3
```

Relevance can be graded. For TREC-DL 2019, labels are `0-3`; use `--relevance-threshold 2` when you want Precision@k and MRR@k to count only highly relevant or perfectly relevant passages. nDCG@k uses the graded values directly.

## Run Tests

Install the test dependency if needed:

```bash
python -m pip install ".[dev]"
```

Then run:

```bash
python -m pytest
```

## How This Maps to the Proposal

This code implements the baseline system described in section 4.1 of the report: BM25 retrieval over a fixed corpus. MS MARCO/TREC-DL 2019 gives the project real passage-ranking queries and graded relevance labels, creating a stronger comparison point for the remaining experiments:

- BM25 + LLM reranking
- BM25 + agent-based iterative refinement
- iteration analysis
- original query vs. refined query analysis

The package is organized so those later systems can reuse the same data loaders, ranking result format, and evaluation metrics.
