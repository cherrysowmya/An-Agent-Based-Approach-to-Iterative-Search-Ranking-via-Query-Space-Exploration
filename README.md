# An Agent-Based Approach to Iterative Search Ranking via Query Space Exploration

This project studies whether search ranking improves when retrieval is treated as an iterative decision-making process instead of a single static ranking step.

Modern search systems often retrieve documents once, rank them, and stop. In real search behavior, though, users inspect results, refine queries, compare candidates, and repeat. This project models that process with an LLM-guided search agent that can evaluate retrieved passages, decide whether the current results are good enough, refine the query when needed, and keep track of the search state across iterations.

The project implements three progressively richer systems from the proposal:

1. **BM25 retrieval baseline**
2. **BM25 + LLM reranking**
3. **BM25 + agent-based iterative refinement**

The main research question is whether LLM scoring and agentic query refinement can improve ranking quality over classical BM25 retrieval.

## System Architecture

The implementation follows the proposal's architecture:

- **Retrieval module:** BM25 retrieves top-k candidate passages.
- **LLM module:** local Ollama `llama3.1:8b` scores relevance, explains scores, and generates refined queries.
- **Planning/control logic:** the agent chooses `refine_query`, `rerank`, or `stop` using average LLM score, score variance, and improvement across iterations.
- **State/memory:** the system stores query history, rankings, scores, actions, and iteration logs.

The agent loop is:

```text
retrieve with BM25
score candidate documents with the LLM
compute average score, variance, and improvement
choose refine_query, rerank, or stop
update state and repeat up to N iterations
```

## Dataset

The experiments use **MS MARCO Passage / TREC-DL 2019 judged** via `ir_datasets`.

The raw MS MARCO passage collection is large, so the converter materializes a smaller local judged-document pool in the CSV format used by the project:

```text
data/msmarco_trec_dl_2019_judged/corpus.csv
data/msmarco_trec_dl_2019_judged/queries.csv
data/msmarco_trec_dl_2019_judged/qrels.csv
```

Optional TREC-DL 2020 judged data can also be generated for validation.

## Setup

Install Python dependencies:

```bash
python3 -m pip install ".[datasets]"
```

Install Ollama and pull the local unpaid model:

```bash
brew install ollama
ollama pull llama3.1:8b
```

If Ollama is not already running:

```bash
ollama serve
```

## Prepare MS MARCO Data

Generate the local judged-pool CSV files:

```bash
python3 -m search_ranker.msmarco \
  --all-trec-dl-judged \
  --out data \
  --purge-msmarco-cache \
  --confirm-purge-msmarco-cache
```

This creates local CSV files for the 2019 and 2020 TREC-DL judged splits. The purge flags remove only the raw MS MARCO `ir_datasets` cache after the smaller CSV exports are written.

## Run the Main Experiment Suite

This command runs the four experiments from the proposal:

```bash
python3 -m search_ranker.suite \
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
  --max-queries 5 \
  --out outputs/msmarco_trec_dl_2019_experiment_suite
```

Remove `--max-queries 5` to run the full TREC-DL 2019 judged query set.

## Experiments

The suite runs:

**Experiment 1: Baseline vs Reranking**

Compares:

```text
BM25
BM25 + LLM reranking
```

Output:

```text
experiment_1_baseline_vs_reranking.csv
```

**Experiment 2: Iterative Agent Performance**

Compares:

```text
BM25
BM25 + LLM reranking
BM25 + agent-based iterative refinement
```

Output:

```text
experiment_2_method_comparison.csv
```

**Experiment 3: Iteration Analysis**

Evaluates ranking metrics across agent iterations.

Output:

```text
experiment_3_iteration_analysis.csv
```

**Experiment 4: Query Refinement Impact**

Compares original-query rankings against refined-query rankings for queries where the agent chose to rewrite the query.

Output:

```text
experiment_4_query_refinement_impact.csv
```

The suite also writes:

```text
suite_summary.json
experiment_1_baseline_bm25/
experiment_1_llm_rerank/
experiment_2_agent/
```

## Metrics

The system reports:

- **Precision@k:** fraction of top-k results that are relevant.
- **MRR@k:** reciprocal rank of the first relevant result.
- **nDCG@k:** graded ranking quality using TREC-DL relevance labels.

For TREC-DL labels, relevance values are `0-3`. The command uses:

```text
--relevance-threshold 2
```

So Precision and MRR count labels `2` and `3` as relevant. nDCG uses the graded labels directly.

## Individual Commands

Run only BM25:

```bash
python3 -m search_ranker.baseline \
  --corpus data/msmarco_trec_dl_2019_judged/corpus.csv \
  --queries data/msmarco_trec_dl_2019_judged/queries.csv \
  --qrels data/msmarco_trec_dl_2019_judged/qrels.csv \
  --top-k 10 \
  --relevance-threshold 2 \
  --out outputs/msmarco_trec_dl_2019_baseline
```

Run BM25 + LLM reranking:

```bash
python3 -m search_ranker.rerank \
  --corpus data/msmarco_trec_dl_2019_judged/corpus.csv \
  --queries data/msmarco_trec_dl_2019_judged/queries.csv \
  --qrels data/msmarco_trec_dl_2019_judged/qrels.csv \
  --bm25-top-k 20 \
  --final-top-k 10 \
  --relevance-threshold 2 \
  --model llama3.1:8b \
  --out outputs/msmarco_trec_dl_2019_llm_rerank
```

Run only the agent:

```bash
python3 -m search_ranker.agent \
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
  --out outputs/msmarco_trec_dl_2019_agent
```

## Output Files

Important agent files:

- `agent_iteration_log.csv`: query history, average score, variance, improvement, action, and next query.
- `agent_candidate_scores.csv`: BM25 rank, BM25 score, LLM score, hybrid score, LLM reasoning, and relevance label.
- `agent_final_rankings.csv`: final selected ranking per query.
- `metrics.json`: aggregate and per-query metrics.

Generated datasets and experiment outputs are intentionally ignored by Git.

## Tests

Install test dependencies:

```bash
python3 -m pip install ".[dev]"
```

Run tests:

```bash
python3 -m pytest
```

## Project Takeaway

The project reframes search ranking as a sequential decision-making problem. BM25 provides a lexical baseline, LLM reranking adds semantic relevance judgment, and the agent loop adds adaptive query refinement and interpretable control logic. This allows the system to show not only whether ranking improves, but also why the system chose to refine, rerank, or stop.
