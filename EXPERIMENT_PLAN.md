# Toollery Experiment Plan

## Current Status (2026-05-26)

Main-paper retrieval and efficiency rows have been run and filled in
`overleaf_toollery_emnlp/acl_latex_skill_6page.tex`.

- SkillRouter full-pool raw, query-augmented, RAGAnything, and Full Toollery
  rows are complete.
- BFCL and BFCL-live raw, query-augmented, RAGAnything, and Full Toollery rows
  are complete.
- BFCL generated-query artifacts are complete for non-live and live splits.
- The full-context LLM reference over complete BFCL/BFCL-live candidate pools is
  not run; the manuscript now marks this explicitly as not run for full
  candidate pools instead of as a pending experiment.
- The optional Query-Augmented SkillRouter row was removed from the manuscript
  because there is no defined local command for an official augmented
  SkillRouter run.

This document maps each remaining experiment to the empty rows in Table 2 and
Table 3 of the current paper draft. Update this file as runs finish or the paper
tables change.

## Global Notes

- Current workspace root:

```text
/Volumes/TXX/projects/toollery_emnlp
```

- The runnable terminal commands below explicitly `cd` into this workspace and
  use `/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python`, so they can be pasted from any terminal location after
  the local virtual environment has been created.
- Do not write to, delete from, or force-rebuild anything under:

```text
outputs/toollery_skillrouter_full_dp/
```

- SkillRouter `BM25 raw` and `SkillRouter raw` are taken directly from the
SkillRouter paper/original results, not rerun in this repo.
- There are three separate embedding configurations in the current scripts:
  - Dense baseline uses `--dense-embedding-*`.
  - RAGAnything / LightRAG uses `--embedding-*`; it now supports
    `--embedding-backend local-hf`.
  - Full Toollery uses `--embedding-*` directly in the command. Use
    `--no-config` when you want the run to ignore JSON config files.
- Dense baselines can use `local-hf` embeddings directly. On a Mac,
`Qwen/Qwen3-Embedding-4B` is slow for full SkillRouter pools. Prefer one of:
  - `Qwen/Qwen3-Embedding-0.6B`
  - `--dense-embedding-max-length 256`
  - larger `--dense-embedding-batch-size` if memory allows
  - `--dense-embedding-local-files-only` after the model is downloaded
- Dense embedding caches are resumable and reusable through
  `--dense-embedding-cache-dir`. Changing batch size no longer invalidates the
  cache.
- RAGAnything also embeds documents. Use `--embedding-backend local-hf` plus
  `--embedding-model`, `--embedding-device`, `--embedding-batch-size`, and
  `--embedding-max-length` to run it with local HuggingFace embeddings.
- Full Toollery also embeds generated or verified proxy queries. Current
  Toollery commands should pass `--embedding-backend local-hf` directly instead
  of relying on a config file.
- Full Toollery should use example queries for the current paper setting. Pass
  `--example-queries-from-relevance` and `--examples-per-skill` explicitly in
  the command, even though example-query conditioning is currently enabled by
  default in the script. This keeps the run provenance unambiguous.

## Execution Order and Parallelism

### Must Run First

These steps produce artifacts that later runs consume.

1. Confirm the source Toollery generated-query files exist:

```text
outputs/toollery_skillrouter/manual_raw_easy.jsonl
outputs/toollery_skillrouter/manual_raw_hard.jsonl
```

If either file is missing, run or finish the corresponding Toollery generation
job first. Do not use files under `outputs/toollery_skillrouter_full_dp/` while
that background job is incomplete, and never read `.partial.jsonl` files as
final generated-query artifacts.

2. Normalize SkillRouter generated queries.

Run the two normalize commands in the next section before any SkillRouter
query-augmented baseline:

```text
Terminal 2
Terminal 3
Terminal 5
Terminal 6
```

The easy and hard normalize commands are independent and can run in parallel.

3. For Table 3 query-augmented baselines, generate or export BFCL tool-level
generated queries first:

```text
outputs/experiments_v2/bfcl/generated_queries/nonlive.jsonl
outputs/experiments_v2/bfcl_live/generated_queries/live.jsonl
```

Until these files exist, do not run the Table 3 query-augmented command block.
Use the BFCL generated-query commands in the Table 3 section below to create
these files with the same LLM proxy-query prompt used by the SkillRouter
Toollery script.

### Cache Reuse Rules

- Dense baseline caches are written under `--dense-embedding-cache-dir`. For
  `local-hf` and OpenAI-compatible dense embeddings, the runner uses a global
  per-document cache (`global_documents.jsonl`) and each sample slices vectors
  from that cache. This avoids re-embedding the same tool document across BFCL
  candidate pools. For the hashing TF-IDF backend, caching remains pool-scoped
  because IDF depends on the candidate pool.
- RAGAnything caches are written under `--rag-working-dir`. They are reusable
  only for the same raw/query-augmented document set and embedding config.
- Full Toollery proxy-query embeddings are written to `--manual-embeddings-out`
  under the selected `--output-dir`.
- Do not launch two identical runs that write the same output directory and the
  same cache files at the same time.
- Changing only batch size should not invalidate Dense or Toollery embedding
  caches. Changing model, max length, pooling, backend, or document content
  should create or require a different cache.
- Keep Qwen3-0.6B and Qwen3-4B caches in separate directories, as the commands
  below do with `qwen3_06b` and `qwen3_4b` suffixes.

### Can Run In Parallel After Prerequisites

These are logically independent because they write different output/cache
locations:

```text
Terminal 1: SkillRouter Dense raw
Terminal 4: SkillRouter RAGAnything raw
Terminal 7: BFCL non-live BM25 + Dense raw
Terminal 8: BFCL live BM25 + Dense raw
Terminal 9: BFCL non-live RAGAnything raw
Terminal 10: BFCL live RAGAnything raw
```

After SkillRouter generated queries are normalized, these can also run in
parallel:

```text
Terminal 2: SkillRouter query-augmented BM25 + Dense, easy
Terminal 3: SkillRouter query-augmented BM25 + Dense, hard
Terminal 5: SkillRouter query-augmented RAGAnything, easy
Terminal 6: SkillRouter query-augmented RAGAnything, hard
```

After BFCL generated queries exist, BFCL non-live and BFCL-live
query-augmented runs can run in parallel if they write to separate output,
dense-cache, and RAG-cache directories.

### Recommended Local-HF Scheduling

The commands above are logically parallel, but local HuggingFace embedding on
Mac MPS is the bottleneck. For stability and throughput:

- Run at most one or two local-hf embedding-heavy jobs at the same time on MPS.
- Prefer finishing one large SkillRouter embedding cache before starting many
  other embedding jobs.
- BM25-only work is cheap and can run anytime, but several commands combine BM25
  with Dense or RAGAnything, so the combined command is still embedding-heavy.
- If a run is interrupted, rerun the same command with the same cache/output
  paths. It should resume or reuse the saved embeddings.

### Suggested Run Waves

Wave 0, prerequisite artifacts:

```text
Normalize SkillRouter generated queries.
Generate/export BFCL generated queries if you want Table 3 query-augmented rows.
```

Wave 1, first large caches:

```text
Terminal 1: SkillRouter Dense raw
Terminal 7 or 8: one BFCL Dense raw job
```

Wave 2, SkillRouter query-augmented retrieval:

```text
Terminal 2 and Terminal 3, preferably after Terminal 1 has warmed/downloaded the model.
```

Wave 3, RAGAnything local-hf:

```text
Terminal 4, then Terminal 5/6, plus Terminal 9/10 as resources allow.
```

Wave 4, Full Toollery:

```text
Run the Toollery command after confirming LLM environment variables and after
deciding whether to reuse existing `outputs/toollery_skillrouter/*` results.
```

Wave 5, paper aggregation:

```text
Aggregate Table 4 token/latency compression.
Aggregate offline construction metrics.
```

These aggregation steps should run after the relevant retrieval/Toollery jobs
finish. They are read-only over completed artifacts and can run in parallel with
unrelated new experiments, but should not read `.partial.jsonl` files.

## Manuscript Cross-Check: Items Beyond Table 2 and Table 3

The current main draft `overleaf_toollery_emnlp/acl_latex_skill_6page.tex`
contains experimental claims and tables beyond the Table 2/Table 3 retrieval
rows. Track these explicitly so they are either run, cited, reused, or removed
from the paper.

| Manuscript item | Status in this plan | Action needed |
|---|---|---|
| Figure 1 candidate-pool degradation curve | Not mapped to a run | Reuse old figure/data or record the script/artifact that generated `figs/figure1.jpg`; if refreshed, rerun ScaleTool-style pool-size analysis. |
| Table 2 Raw Dense `Qwen3-Emb-8B` dagger row | Mismatch with Terminal 1 | Either cite SkillRouter's Qwen3-Emb-8B encoder-only row and mark Terminal 1 as optional local dense sanity check, or change the paper row label to the local-hf model actually run. |
| Table 2 Query-Augmented SkillRouter row | Marked optional, no command | If kept in the table, implement/run an official SkillRouter or SkillRouter-style run over augmented skill text; otherwise mark N/A or remove the row. |
| Table 3 ScaleTool-style atomic retrieval | Only BFCL/BFCL-live baseline commands are listed | If Table 3 is meant to include ScaleTool-HammerBench and ScaleTool-xLAM retrieval metrics, add raw/query-augmented BM25, Dense, RAGAnything, and Full Toollery retrieval runs for those candidate pools. Otherwise narrow the Table 3 caption to BFCL-derived retrieval. |
| Table 3 xLAM/HammerBench retrieval rows | Not separately mapped | Same as above: the paper mentions xLAM/HammerBench end-to-end results, but Table 3 retrieval metrics need separate gold retrieval runs if reported there. |
| Table 4 Full-context LLM reference | Missing concrete command | Add `full_context_llm_raw` runs for feasible pool sizes, or compute from existing end-to-end prompt baselines; this is the denominator for token/latency compression. |
| Table 4 RAGAnything subset rows | Partially covered by retrieval commands, not efficiency aggregation | Ensure RAGAnything runs log `prompt_tokens`, `llm_latency_ms`, `total_latency_ms`; then aggregate into Table 4. |
| Table 4 SkillRouter subset rows | Not covered if SkillRouter raw is only cited | Need official/SkillRouter-style local runs with latency/token logging, or remove/mark N/A for SkillRouter rows in Table 4. Cited paper metrics are not enough for compression measurements. |
| Table 4 Query-Augmented SkillRouter row | No command | Same dependency as Query-Augmented SkillRouter in Table 2; keep only if augmented SkillRouter can run locally. |
| Table 4 Full Toollery efficiency | Retrieval command exists, aggregation missing | Add/execute an aggregator over Toollery predictions and final-selector logs to compute candidates-to-LLM, prompt tokens, LLM latency, total latency, and compression. |
| Offline construction metrics | Mentioned in text, no aggregation command | Add/execute a stats extractor for generated queries per skill, verification acceptance rate, verified queries per skill, zero-query skill rate, construction latency, and indexing cost. |
| End-to-end HammerBench table | Reused old table only | Record source artifacts/scripts for `table_scaletool_hb.tex`; rerun only if model names/results need updating. |
| End-to-end xLAM table | Reused old table only | Record source artifacts/scripts for `table_scaletool_xlam.tex`; rerun only if model names/results need updating. |
| End-to-end BFCL non-live table | Reused old table only | Record source artifacts/scripts for `table_ablation_backborn_bfcl.tex`; rerun only if model names/results need updating. |
| End-to-end BFCL live table | Reused old table only | Record source artifacts/scripts for `table_bfcl_live_multiple.tex`; rerun only if model names/results need updating. |
| Candidate-pool sizes `5000` and `10000` in text | Not covered by current tables/commands | Current old tables go up to `1000`. Either add larger-pool retrieval-only or retrieval-plus-subset runs, or revise the paper sentence to match available pool sizes. |

Priority fixes before filling the paper:

1. Decide whether Table 2 Raw Dense is cited from SkillRouter or rerun locally.
2. Decide whether Query-Augmented SkillRouter and Table 4 SkillRouter rows stay
   in the paper. If they stay, they need implementation/runs.
3. Add Table 4 aggregation, because token/latency compression is a central
   claim and is not covered by the current retrieval-only plan.
4. Add offline construction stats extraction or soften the text that promises
   those statistics.
5. Either add ScaleTool/xLAM/HammerBench retrieval baseline runs for Table 3 or
   narrow Table 3 to BFCL-derived retrieval and keep HammerBench/xLAM as
   end-to-end tables only.

## Table 4: Token and Latency Compression

Table 4 in the main draft is not a retrieval-quality table. It measures how much
candidate compression reduces the final selector prompt and latency. It should
be produced after the main retrieval runs, because it depends on the candidate
subsets emitted by each method.

### Rows to Fill

| Table 4 row | Required source | Current status |
|---|---|---|
| Reference / Full-context LLM | Full candidate pool sent to final LLM | BFCL runner has `full_context_llm_raw`; SkillRouter runner needs per-sample efficiency logging if used. |
| Raw Baseline / RAGAnything subset | RAGAnything raw top-k candidates + final selector prompt stats | Retrieval commands exist; need final-selector efficiency aggregation. |
| Raw Baseline / SkillRouter subset | SkillRouter or SkillRouter-style raw top-k candidates + final selector prompt stats | Not covered by cited paper values; needs local run or remove/mark N/A. |
| Query-Augmented / RAGAnything + generated queries | RAGAnything query-augmented top-k candidates + final selector prompt stats | Retrieval commands exist after generated queries are normalized; need final-selector efficiency aggregation. |
| Query-Augmented / SkillRouter + generated queries | Augmented SkillRouter top-k candidates + final selector prompt stats | Needs implementation/run if the row stays in the paper. |
| Full Toollery / Dense verified-query retrieval | Full Toollery top-k candidates + final selector prompt stats | Toollery command exists; need final-selector efficiency aggregation. |

### Metrics to Aggregate

For each row, report:

```text
candidate_pool_size
candidates_to_llm
prompt_tokens
completion_tokens
retrieval_latency_ms
rerank_latency_ms
llm_latency_ms
total_latency_ms
estimated_cost_per_1k_queries
token_compression
latency_compression
```

The paper table currently displays only:

```text
candidate_pool_size
candidates_to_llm
prompt_tokens
llm_latency
total_latency
token_compression
```

Keep the full metric set in JSON/CSV artifacts and show the compact subset in
the paper.

### Compression Formulas

For a fixed benchmark split and candidate pool size:

```text
TokenCompression =
  AvgPromptTokens_full_context / AvgPromptTokens_method

LatencyCompression =
  AvgLLMLatency_full_context / AvgLLMLatency_method
```

If full-context LLM is infeasible for a large pool, use the largest feasible
full-context pool as the reference and mark the corresponding paper cell with a
note.

### Concrete Runs Needed

BFCL / atomic track, full-context reference where feasible:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_bfcl_baselines.py \
  --scaled-data bfcl_scaled_data.jsonl \
  --methods full_context_llm_raw \
  --selector llm \
  --top-k 3 \
  --output-dir outputs/experiments_v2/bfcl/full_context_llm
```

BFCL live full-context reference:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_bfcl_baselines.py \
  --scaled-data bfcl_scaled_data_live.jsonl \
  --methods full_context_llm_raw \
  --selector llm \
  --top-k 3 \
  --output-dir outputs/experiments_v2/bfcl_live/full_context_llm
```

RAGAnything and Dense retrieval commands in Terminal 4--10 already estimate
prompt tokens for the retrieved subset. However, they currently do not call the
final LLM selector unless the runner is extended to do retrieval-plus-selector
evaluation. To fill `llm_latency_ms` rather than only `retrieval_latency_ms`,
add or run a final-selector evaluation mode that takes saved top-k candidates
and calls the same final selector as Full Toollery.

Recommended aggregation script to add:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/aggregate_table4_compression.py \
  --full-context outputs/experiments_v2/bfcl/full_context_llm/full_context_llm_raw/predictions.jsonl \
  --runs \
    outputs/experiments_v2/bfcl/rag_raw_qwen3_06b/raganything_raw/predictions.jsonl \
    outputs/experiments_v2/bfcl/query_augmented_qwen3_06b/raganything_query_augmented/predictions.jsonl \
    outputs/experiments_v2/bfcl/full_toollery_qwen3_06b/predictions.jsonl \
  --out outputs/experiments_v2/tables/table4_bfcl_compression.json \
  --tex-out overleaf_toollery_emnlp/table_token_latency_compression_values.tex
```

The script should compute per-method averages and compression ratios, then
export both machine-readable JSON and a TeX-friendly value file. Repeat the same
aggregation for SkillRouter only if local SkillRouter subset runs have
efficiency logs. SkillRouter paper citation values alone are not sufficient for
Table 4 because they do not provide our prompt-token or latency measurements.

### Table 4 Dependencies and Parallelism

- Full-context reference must finish before compression ratios can be computed.
- Retrieval/subset methods can run before or after full-context reference.
- Aggregation is read-only and can run after all required inputs for a benchmark
  are complete.
- Do not compare compression across different candidate pools, different final
  selectors, or different prompt templates.

## Offline Construction Metrics

The main draft says we report construction statistics for Full Toollery. These
metrics describe the cost and coverage of building the verified query index;
they are not a self-verification ablation.

### Metrics to Report

For each benchmark/tier:

```text
selected_skills
generated_candidate_queries
verified_queries_retained
verification_acceptance_rate
avg_generated_queries_per_skill
avg_verified_queries_per_skill
zero_verified_query_skill_count
zero_verified_query_skill_rate
offline_generation_verification_latency_ms
embedding_indexing_latency_ms
offline_total_latency_ms
estimated_generation_verification_cost
estimated_embedding_cost
```

For SkillRouter Toollery, the basic count fields can be extracted from:

```text
manual_raw_{tier}.jsonl
manual_verified_{tier}.json
summary.json
embeddings/manual_embeddings_{tier}.json
```

For the current v2 local-hf command, these paths are under:

```text
outputs/experiments_v2/skillrouter/full_toollery_gt_qwen3_06b/
```

Existing `outputs/toollery_skillrouter/*` artifacts can be reused for counts if
they are complete, but keep them separate from new v2 runs in the paper notes.
Do not read or summarize:

```text
outputs/toollery_skillrouter_full_dp/**/*.partial.jsonl
```

### Concrete Aggregation Command to Add

Recommended script:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/summarize_offline_construction.py \
  --run-dir outputs/experiments_v2/skillrouter/full_toollery_gt_qwen3_06b \
  --tiers easy hard \
  --manual-raw manual_raw_{tier}.jsonl \
  --manual manual_verified_{tier}.json \
  --manual-embeddings embeddings/manual_embeddings_{tier}.json \
  --summary summary.json \
  --out outputs/experiments_v2/tables/offline_construction_skillrouter.json \
  --tex-out overleaf_toollery_emnlp/table_offline_construction_values.tex
```

The script should:

1. Count raw generated query rows per skill from `manual_raw_{tier}.jsonl`.
2. Count retained verified queries per skill from `manual_verified_{tier}.json`.
3. Compute acceptance rate as `verified_queries_retained /
   generated_candidate_queries`.
4. Compute zero-verified-query rate over the selected skills in the run summary.
5. Read embedding cache metadata/counts from `manual_embeddings_{tier}.json` if
   present.
6. Include latency/cost fields from the run summary if available; if the current
   runner does not log them yet, output `null` and add runner instrumentation
   before filling those paper cells.

### Runner Instrumentation Still Needed

Current Toollery summaries already include selected skill counts, raw row counts,
verified manual entries, artifact paths, and retrieval metrics. To fully satisfy
the paper text, add timing/cost logging around:

```text
manual generation + verification
manual embedding/index construction
online retrieval evaluation
```

If timing/cost instrumentation is not added, the paper should report only
coverage/count statistics and avoid claiming construction latency or indexing
cost values.

## Table 2: SkillRouter / SkillBench-Compatible

| Setting | Method | Source / Experiment | Command or Notes |
|---|---|---|---|
| Raw Baseline | BM25 | Do not run; take from SkillRouter paper | Use the BM25 row from SkillRouter Table 2 |
| Raw Baseline | Dense | Run our general dense baseline | Terminal 1 |
| Raw Baseline | RAGAnything | Run | Terminal 4 |
| Raw Baseline | SkillRouter | Do not run; take from SkillRouter paper | Use SR-Emb / SR-Rank depending on final comparison |
| Query-Augmented Baseline | BM25 | Run | Terminal 2 easy + Terminal 3 hard |
| Query-Augmented Baseline | Dense | Run | Terminal 2 easy + Terminal 3 hard |
| Query-Augmented Baseline | RAGAnything | Run | Terminal 5 easy + Terminal 6 hard |
| Query-Augmented Baseline | SkillRouter | Optional / currently not run | Only if official SkillRouter supports extended skill text; otherwise mark N/A or SkillRouter-style |
| Toollery | Dense | Reuse existing or rerun into v2 directory | `outputs/toollery_skillrouter/*` or Full Toollery rerun |

### Normalize SkillRouter Generated Queries

Run these before query-augmented SkillRouter baselines:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/normalize_generated_queries.py \
  --source outputs/toollery_skillrouter/manual_raw_easy.jsonl \
  --out outputs/experiments_v2/skillrouter/generated_queries/easy.jsonl


/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/normalize_generated_queries.py \
  --source outputs/toollery_skillrouter/manual_raw_hard.jsonl \
  --out outputs/experiments_v2/skillrouter/generated_queries/hard.jsonl
```

### Terminal 1: Table 2 - Raw Baseline / Dense

Fast recommended version with smaller embedding model:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers easy hard \
  --methods dense_raw \
  --dense-embedding-backend local-hf \
  --dense-embedding-model Qwen/Qwen3-Embedding-0.6B \
  --dense-embedding-device mps \
  --dense-embedding-batch-size 16 \
  --dense-embedding-max-length 256 \
  --top-k 50 \
  --output-dir outputs/experiments_v2/skillrouter/dense_raw_qwen3_06b \
  --dense-embedding-cache-dir outputs/experiments_v2/skillrouter/dense_embedding_cache_qwen3_06b
```

Slower 4B version:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers easy hard \
  --methods dense_raw \
  --dense-embedding-backend local-hf \
  --dense-embedding-model Qwen/Qwen3-Embedding-4B \
  --dense-embedding-device mps \
  --dense-embedding-batch-size 8 \
  --dense-embedding-max-length 256 \
  --dense-embedding-local-files-only \
  --top-k 50 \
  --output-dir outputs/experiments_v2/skillrouter/dense_raw_qwen3_4b \
  --dense-embedding-cache-dir outputs/experiments_v2/skillrouter/dense_embedding_cache_qwen3_4b
```

### Terminal 2: Table 2 - Query-Augmented Baseline / BM25 + Dense, Easy

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers easy \
  --generated-queries outputs/experiments_v2/skillrouter/generated_queries/easy.jsonl \
  --methods bm25_query_augmented,dense_query_augmented \
  --dense-embedding-backend local-hf \
  --dense-embedding-model Qwen/Qwen3-Embedding-0.6B \
  --dense-embedding-device mps \
  --dense-embedding-batch-size 16 \
  --dense-embedding-max-length 256 \
  --top-k 50 \
  --output-dir outputs/experiments_v2/skillrouter/query_augmented_easy_qwen3_06b \
  --dense-embedding-cache-dir outputs/experiments_v2/skillrouter/dense_embedding_cache_qwen3_06b
```

### Terminal 3: Table 2 - Query-Augmented Baseline / BM25 + Dense, Hard

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers hard \
  --generated-queries outputs/experiments_v2/skillrouter/generated_queries/hard.jsonl \
  --methods bm25_query_augmented,dense_query_augmented \
  --dense-embedding-backend local-hf \
  --dense-embedding-model Qwen/Qwen3-Embedding-0.6B \
  --dense-embedding-device mps \
  --dense-embedding-batch-size 16 \
  --dense-embedding-max-length 256 \
  --top-k 50 \
  --output-dir outputs/experiments_v2/skillrouter/query_augmented_hard_qwen3_06b \
  --dense-embedding-cache-dir outputs/experiments_v2/skillrouter/dense_embedding_cache_qwen3_06b
```

### Terminal 4: Table 2 - Raw Baseline / RAGAnything

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers easy hard \
  --methods raganything_raw \
  --backend raganything \
  --embedding-backend local-hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device mps \
  --embedding-batch-size 16 \
  --embedding-max-length 256 \
  --top-k 50 \
  --output-dir outputs/experiments_v2/skillrouter/rag_raw_qwen3_06b \
  --rag-working-dir outputs/experiments_v2/skillrouter/rag_cache_qwen3_06b
```

### Terminal 5: Table 2 - Query-Augmented Baseline / RAGAnything, Easy

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers easy \
  --generated-queries outputs/experiments_v2/skillrouter/generated_queries/easy.jsonl \
  --methods raganything_query_augmented \
  --backend raganything \
  --embedding-backend local-hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device mps \
  --embedding-batch-size 16 \
  --embedding-max-length 256 \
  --top-k 50 \
  --output-dir outputs/experiments_v2/skillrouter/rag_query_augmented_easy_qwen3_06b \
  --rag-working-dir outputs/experiments_v2/skillrouter/rag_cache_qwen3_06b
```

### Terminal 6: Table 2 - Query-Augmented Baseline / RAGAnything, Hard

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers hard \
  --generated-queries outputs/experiments_v2/skillrouter/generated_queries/hard.jsonl \
  --methods raganything_query_augmented \
  --backend raganything \
  --embedding-backend local-hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device mps \
  --embedding-batch-size 16 \
  --embedding-max-length 256 \
  --top-k 50 \
  --output-dir outputs/experiments_v2/skillrouter/rag_query_augmented_hard_qwen3_06b \
  --rag-working-dir outputs/experiments_v2/skillrouter/rag_cache_qwen3_06b
```

### Table 2 - Toollery / Dense

Reusable existing results:

```text
outputs/toollery_skillrouter/*
```

Terminal 11A: Optional unified rerun, cheaper diagnostic / gt-related scope:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_toollery.py \
  --config skillrouter_toollery_deepseek_config.example.json \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers easy hard \
  --manual-scope gt-related \
  --example-queries-from-relevance \
  --examples-per-skill 3 \
  --embedding-backend local-hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device mps \
  --embedding-batch-size 16 \
  --embedding-max-length 256 \
  --output-dir outputs/experiments_v2/skillrouter/full_toollery_gt_qwen3_06b \
  --top-k 50 \
  --proxy-top-k 1000
```

Terminal 11B: Full-pool example-conditioned run, preferred if this row is used
as a main SkillRouter/SkillBench-compatible result:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_skillrouter_toollery.py \
  --config skillrouter_toollery_deepseek_config.example.json \
  --skillrouter-root /Volumes/TXX/projects/skillrouter/SkillRouter \
  --tiers easy hard \
  --manual-scope full-pool \
  --example-queries-from-relevance \
  --examples-per-skill 3 \
  --proxy-queries-per-skill 3 \
  --embedding-backend local-hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device mps \
  --embedding-batch-size 16 \
  --embedding-max-length 256 \
  --output-dir outputs/experiments_v2/skillrouter/full_toollery_fullpool_examples_qwen3_06b \
  --manual-raw-out manual_raw_{tier}.jsonl \
  --manual-out manual_verified_{tier}.json \
  --manual-embeddings-out embeddings/manual_embeddings_{tier}.json \
  --predictions-out retrieval/{tier}.json \
  --metrics-out metrics/{tier}.json \
  --summary-out summary.json \
  --top-k 50 \
  --proxy-top-k 1000 \
  --llm-workers 8 \
  --llm-batch-size 8
```

Because this command uses `skillrouter_toollery_deepseek_config.example.json`,
make sure the DeepSeek key is available through `DEEPSEEK_API_KEY` or update the
config with the intended provider credentials.

For example, before Terminal 11A/11B:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

export LLM_API_KEY=YOUR_PROVIDER_KEY
export LLM_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-v4-pro
```

If a Toollery run previously failed with an empty verified manual because the
LLM key was missing, rerun the same command with `--force-rebuild` once after
setting the environment variables. This discards the generated `generation_error`
rows and rebuilds `manual_raw_{tier}.jsonl` / `manual_verified_{tier}.json`.

## Table 3: Atomic Tool / Atomic Skill

| Setting | Method | Source / Experiment | Command or Notes |
|---|---|---|---|
| Raw Baseline | BM25 | Run | Terminal 7 + Terminal 8 |
| Raw Baseline | Dense | Run | Terminal 7 + Terminal 8 |
| Raw Baseline | RAGAnything | Run | Terminal 9 + Terminal 10 |
| Query-Augmented Baseline | BM25 | Cannot run yet | Needs BFCL / atomic tool generated queries |
| Query-Augmented Baseline | Dense | Cannot run yet | Needs BFCL / atomic tool generated queries |
| Query-Augmented Baseline | RAGAnything | Cannot run yet | Needs BFCL / atomic tool generated queries |
| Toollery | Dense | Existing old results reusable; unified reruns should use local-hf | Old tables + `bfcl_scaled_toollery_predictions.jsonl`; any rerun should pass `--embedding-backend local-hf` and the same Qwen embedding args directly |

### Terminal 7: Table 3 - Raw Baseline / BM25 + Dense, BFCL Non-Live

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_bfcl_baselines.py \
  --scaled-data bfcl_scaled_data.jsonl \
  --methods bm25_raw,dense_raw \
  --dense-embedding-backend local-hf \
  --dense-embedding-model Qwen/Qwen3-Embedding-0.6B \
  --dense-embedding-device mps \
  --dense-embedding-batch-size 16 \
  --dense-embedding-max-length 256 \
  --top-k 3 \
  --output-dir outputs/experiments_v2/bfcl/raw_baselines_qwen3_06b \
  --dense-embedding-cache-dir outputs/experiments_v2/bfcl/dense_embedding_cache_qwen3_06b
```

### Terminal 8: Table 3 - Raw Baseline / BM25 + Dense, BFCL Live

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_bfcl_baselines.py \
  --scaled-data bfcl_scaled_data_live.jsonl \
  --methods bm25_raw,dense_raw \
  --dense-embedding-backend local-hf \
  --dense-embedding-model Qwen/Qwen3-Embedding-0.6B \
  --dense-embedding-device mps \
  --dense-embedding-batch-size 16 \
  --dense-embedding-max-length 256 \
  --top-k 3 \
  --output-dir outputs/experiments_v2/bfcl_live/raw_baselines_qwen3_06b \
  --dense-embedding-cache-dir outputs/experiments_v2/bfcl_live/dense_embedding_cache_qwen3_06b
```

### Terminal 9: Table 3 - Raw Baseline / RAGAnything, BFCL Non-Live

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_bfcl_baselines.py \
  --scaled-data bfcl_scaled_data.jsonl \
  --methods raganything_raw \
  --backend raganything \
  --embedding-backend local-hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device mps \
  --embedding-batch-size 16 \
  --embedding-max-length 256 \
  --top-k 3 \
  --output-dir outputs/experiments_v2/bfcl/rag_raw_qwen3_06b \
  --rag-working-dir outputs/experiments_v2/bfcl/rag_cache_qwen3_06b
```

### Terminal 10: Table 3 - Raw Baseline / RAGAnything, BFCL Live

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_bfcl_baselines.py \
  --scaled-data bfcl_scaled_data_live.jsonl \
  --methods raganything_raw \
  --backend raganything \
  --embedding-backend local-hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device mps \
  --embedding-batch-size 16 \
  --embedding-max-length 256 \
  --top-k 3 \
  --output-dir outputs/experiments_v2/bfcl_live/rag_raw_qwen3_06b \
  --rag-working-dir outputs/experiments_v2/bfcl_live/rag_cache_qwen3_06b
```

### Table 3 - Query-Augmented Baselines / BM25, Dense, RAGAnything

Currently missing:

```text
outputs/experiments_v2/bfcl/generated_queries/*.jsonl
```

Need to generate or export BFCL tool-level generated queries first.

Generate BFCL non-live tool-level proxy queries:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/generate_bfcl_generated_queries.py \
  --config skillrouter_toollery_deepseek_config.example.json \
  --data /Volumes/TXX/projects/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multiple.json \
  --answers /Volumes/TXX/projects/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_multiple.json \
  --out outputs/experiments_v2/bfcl/generated_queries/nonlive.jsonl \
  --manual-raw-out outputs/experiments_v2/bfcl/generated_queries/manual_raw_nonlive.jsonl \
  --manual-out outputs/experiments_v2/bfcl/generated_queries/manual_nonlive.json \
  --summary-out outputs/experiments_v2/bfcl/generated_queries/summary_nonlive.json \
  --proxy-queries-per-tool 3 \
  --examples-per-tool 3 \
  --llm-workers 8 \
  --llm-batch-size 8
```

Generate BFCL live tool-level proxy queries:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/generate_bfcl_generated_queries.py \
  --config skillrouter_toollery_deepseek_config.example.json \
  --data /Volumes/TXX/projects/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_multiple.json \
  --answers /Volumes/TXX/projects/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_live_multiple.json \
  --out outputs/experiments_v2/bfcl_live/generated_queries/live.jsonl \
  --manual-raw-out outputs/experiments_v2/bfcl_live/generated_queries/manual_raw_live.jsonl \
  --manual-out outputs/experiments_v2/bfcl_live/generated_queries/manual_live.json \
  --summary-out outputs/experiments_v2/bfcl_live/generated_queries/summary_live.json \
  --proxy-queries-per-tool 3 \
  --examples-per-tool 3 \
  --llm-workers 8 \
  --llm-batch-size 8
```

After that, run non-live:

```bash
cd /Volumes/TXX/projects/toollery_emnlp

/Users/txx1220/Projects/toollery_uptodate/.venv/bin/python scripts/run_bfcl_baselines.py \
  --scaled-data bfcl_scaled_data.jsonl \
  --generated-queries outputs/experiments_v2/bfcl/generated_queries/nonlive.jsonl \
  --methods bm25_query_augmented,dense_query_augmented,raganything_query_augmented \
  --dense-embedding-backend local-hf \
  --dense-embedding-model Qwen/Qwen3-Embedding-0.6B \
  --dense-embedding-device mps \
  --dense-embedding-batch-size 16 \
  --dense-embedding-max-length 256 \
  --backend raganything \
  --embedding-backend local-hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device mps \
  --embedding-batch-size 16 \
  --embedding-max-length 256 \
  --top-k 3 \
  --output-dir outputs/experiments_v2/bfcl/query_augmented_qwen3_06b \
  --dense-embedding-cache-dir outputs/experiments_v2/bfcl/dense_embedding_cache_qwen3_06b \
  --rag-working-dir outputs/experiments_v2/bfcl/rag_cache_qwen3_06b
```

For live, replace:

```text
bfcl_scaled_data.jsonl
outputs/experiments_v2/bfcl/generated_queries/nonlive.jsonl
outputs/experiments_v2/bfcl/query_augmented_qwen3_06b
```

with:

```text
bfcl_scaled_data_live.jsonl
outputs/experiments_v2/bfcl_live/generated_queries/live.jsonl
outputs/experiments_v2/bfcl_live/query_augmented_qwen3_06b
```

## One-Line Summary

- Table 2 gaps: run SkillRouter `Dense raw`, query-augmented BM25/Dense,
  RAGAnything raw/query-augmented, and Toollery.
- Table 3 gaps: run BFCL/BFCL-live raw BM25/Dense/RAGAnything first; query-
  augmented runs require BFCL generated queries.
- SkillRouter `BM25 raw` and `SkillRouter raw` are cited from the SkillRouter
  paper and are not rerun.
