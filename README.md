# Toollery Python Flow

This repository contains an end-to-end Python implementation of the workflow described in `Toollery: Scaling LLM Agents to Thousands of Tools`. The core idea is to split large-scale tool selection into two stages:

1. Offline stage: generate diverse proxy queries for each tool and filter noisy samples with round-trip verification.
2. Online stage: retrieve proxy queries by intent-to-intent matching, aggregate them into top-k tool candidates, and pass the compact candidate set to the final selector.

The default implementation uses dependency-free heuristic teacher, verifier, and selector components so the full pipeline can run locally. In production, `OpenAICompatibleLLM` can be used as the teacher, verifier, or final selector.

## Project Structure

- `toollery/manual.py`: implements Algorithm 1 from the paper, synthesizing and verifying proxy queries from tool metadata.
- `toollery/retrieval.py`: performs proxy-query retrieval, tool-level score aggregation, and compact candidate-set construction.
- `toollery/pipeline.py`: runs the online inference flow by retrieving candidate tools and producing a tool call.
- `toollery/scaletool.py`: provides a ScaleTool-style evaluation under candidate-set growth.
- `toollery/cli.py`: exposes the workflow through a command-line interface.
- `examples/`: contains runnable tool definitions, a generated manual, and sample evaluation cases.

## Quick Start

```bash
python -m toollery.cli build-manual \
  --tools examples/tools.json \
  --out examples/manual.json \
  --queries-per-tool 8 \
  --distractors 4
```

```bash
python -m toollery.cli query \
  --tools examples/tools.json \
  --manual examples/manual.json \
  --q "Should I pack an umbrella for Shanghai tomorrow?" \
  --top-k 3
```

```bash
python -m toollery.cli benchmark \
  --tools examples/tools.json \
  --manual examples/manual.json \
  --cases examples/cases.json \
  --sizes 2,3,5,6 \
  --top-k 3
```

Run Toollery over a BFCL JSONL dataset:

```bash
python -m toollery.cli bfcl-batch \
  --data /path/to/BFCL_v4_multiple.json \
  --answers /path/to/possible_answer/BFCL_v4_multiple.json \
  --out bfcl_toollery_predictions.jsonl \
  --top-k 3
```

`bfcl-batch` builds a proxy-query manual from the functions in the selected BFCL split, runs each row with its own candidate function set, writes one prediction per line, and reports tool-name accuracy when `--answers` is provided.

Run ScaleTool-style BFCL expansion before Toollery:

```bash
python -m toollery.cli bfcl-scaletool \
  --data /path/to/BFCL_v4_multiple.json \
  --answers /path/to/possible_answer/BFCL_v4_multiple.json \
  --out bfcl_scaled_toollery_predictions.jsonl \
  --manual-out bfcl_toollery_manual.json \
  --scaled-data-out bfcl_scaled_data.jsonl \
  --sizes 2,5,10,20,50,100 \
  --top-k 3
```

`bfcl-scaletool` ignores the original per-row candidate set size, keeps the ground-truth tool, samples distractor tools from the full BFCL tool pool, and evaluates Toollery under each requested candidate-pool size. Each output row includes both `scale_candidate_pool` and the Toollery-retrieved `retrieved_tools`.

When `--manual-out` or `--scaled-data-out` points to an existing file, the CLI reuses that artifact by default. Add `--force-rebuild` to regenerate the manual and scaled dataset.

Run the LLM-generated skill benchmark experiment:

```bash
python -m toollery.cli skill-scaletool \
  --skills-root data/openharmony-skills \
  --tools-out skill_tools.json \
  --raw-benchmark-out skill_benchmark_raw.jsonl \
  --benchmark-out skill_benchmark_verified.jsonl \
  --manual-out skill_toollery_manual.json \
  --scaled-data-out skill_scaled_data.jsonl \
  --out skill_scaled_toollery_predictions.jsonl \
  --queries-per-skill 5 \
  --sizes 2,5,10,20,50,100,200 \
  --top-k 3
```

`skill-scaletool` parses every `SKILL.md` directory into a tool, uses an OpenAI-compatible LLM to generate realistic user requests, verifies each generated request with an LLM round-trip check, scales candidate skill pools, and evaluates Toollery. It reuses existing `--tools-out`, `--raw-benchmark-out`, `--benchmark-out`, `--manual-out`, and `--scaled-data-out` artifacts unless `--force-rebuild` is provided.

You can also put the same settings in a JSON config file:

```bash
python -m toollery.cli skill-scaletool --config skill_scaletool_config.example.json
```

The `openai`, `deepseek`, or generic `llm` section can set `api_key`, `api_key_env`, `base_url`, `model`, `timeout`, and `extra_body`. The generic runtime reads `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`; OpenAI-compatible names are still supported for older configs. Command-line flags override config values.

For DeepSeek V4 Pro manual generation, set `DEEPSEEK_API_KEY` and use the DeepSeek example config:

```bash
export DEEPSEEK_API_KEY="..."

python scripts/run_skillrouter_toollery.py \
  --config skillrouter_toollery_deepseek_config.example.json
```

## RAG Baseline Comparison

To compare Toollery against ordinary RAG retrieval, run the baseline over the same scaled candidate-pool file produced by `bfcl-scaletool` or `skill-scaletool`. The baseline indexes only each tool or skill's name and description, retrieves top-k documents for the user query, and reports top-k hit accuracy.

```bash
python -m toollery.cli skill-rag-baseline \
  --scaled-data skill_scaled_data.jsonl \
  --out skill_scaled_rag_predictions.jsonl \
  --compare-to skill_scaled_toollery_predictions.jsonl \
  --compare-out skill_scaled_toollery_vs_rag.jsonl \
  --top-k 3
```

```bash
python scripts/run_bfclscaled_rag.py \
  --config bfclscaled_toollery_deepseek_config.example.json \
  --backend raganything
```

Run the same name-description RAG baseline on SkillRouter Eval Core:

```bash
python scripts/run_skillrouter_rag.py \
  --config skillrouter_toollery_config.json \
  --tiers easy hard \
  --top-k 50 \
  --output-dir outputs/rag_skillrouter
```

This writes `retrieval/{tier}.json` in the same task-id to ranked-skill-ids format as the Toollery SkillRouter run, plus SkillRouter metrics including `Recall@K`, `FullCoverage@K`, `nDCG@K`, and `Hit@1`.

## Revised Experiment Settings

The paper-facing experiment matrix now uses three settings:

- `Raw Baselines`: retrieve over original tool/skill fields only.
- `Query-Augmented Baselines`: append the shared generated queries to each raw document, but do not use Toollery's query-level retrieval formulation.
- `Full Toollery`: generate candidate queries, self-verify them, retrieve over the verified query index, aggregate back to compact tool/skill candidates, and send only top-k compact specs to the final selector.

Do not use the older `Toollery Raw / Shared / Full` naming in paper tables. Do not use the OpenHarmony skill benchmark in the official experiment story.

All new experiment outputs should go under `outputs/experiments_v2/`. Do not write into `outputs/toollery_skillrouter_full_dp/`, which may contain an in-progress full-pool SkillRouter background run and partial embedding caches.

The v2 baseline runners import BM25, dense, and RAG-Anything-compatible retrievers from `toollery.baselines`. The original `toollery.rag_baseline` module and its CLI/script entry points are kept for legacy reproduction of the previous RAG baseline.

Normalize generated queries from a Toollery `manual_raw_*.jsonl` artifact:

```bash
python scripts/normalize_generated_queries.py \
  --source outputs/toollery_skillrouter/manual_raw_easy.jsonl \
  --out outputs/experiments_v2/skillrouter/generated_queries/easy.jsonl
```

Run BFCL raw and query-augmented baselines:

```bash
python scripts/run_bfcl_baselines.py \
  --scaled-data bfcl_scaled_data.jsonl \
  --generated-queries outputs/experiments_v2/skillrouter/generated_queries/easy.jsonl \
  --methods bm25_raw,dense_raw,raganything_raw,bm25_query_augmented,dense_query_augmented,raganything_query_augmented \
  --top-k 3 \
  --dense-embedding-cache-dir outputs/experiments_v2/bfcl/dense_embedding_cache \
  --output-dir outputs/experiments_v2/bfcl/baselines
```

Run SkillRouter raw and query-augmented baselines:

```bash
python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Users/txx1220/Projects/skillrouter/SkillRouter \
  --tiers easy hard \
  --generated-queries outputs/experiments_v2/skillrouter/generated_queries/hard.jsonl \
  --methods bm25_raw,dense_raw,raganything_raw,bm25_query_augmented,dense_query_augmented,raganything_query_augmented \
  --top-k 50 \
  --dense-embedding-cache-dir outputs/experiments_v2/skillrouter/dense_embedding_cache \
  --output-dir outputs/experiments_v2/skillrouter/baselines
```

Dense baseline embeddings and Toollery manual-query embeddings are written as reusable JSONL caches. Interrupted runs leave `.partial.jsonl` files and resume from the completed rows on the next run. Use `--force-rebuild-embeddings` only when you intentionally want to discard the reusable embedding cache.

To use a locally downloaded HuggingFace embedding model for dense baselines, pass `--dense-embedding-backend local-hf` and point `--dense-embedding-model` to the local model directory:

```bash
python scripts/run_skillrouter_baselines.py \
  --skillrouter-root /Users/txx1220/Projects/skillrouter/SkillRouter \
  --tiers easy hard \
  --methods dense_raw \
  --dense-embedding-backend local-hf \
  --dense-embedding-model /path/to/Qwen3-Embedding-4B \
  --dense-embedding-device mps \
  --dense-embedding-local-files-only \
  --top-k 50 \
  --output-dir outputs/experiments_v2/skillrouter/dense_raw_qwen3 \
  --dense-embedding-cache-dir outputs/experiments_v2/skillrouter/dense_embedding_cache_qwen3
```

For Full Toollery runs, set the embedding section in the config:

```json
{
  "embedding": {
    "backend": "local-hf",
    "model_path": "/path/to/Qwen3-Embedding-4B",
    "device": "mps",
    "batch_size": 4,
    "max_length": 512,
    "pooling": "last",
    "local_files_only": true,
    "trust_remote_code": false
  }
}
```

The local backend requires `torch` and `transformers`, or `sentence-transformers`. It never downloads when `local_files_only` is true.

Wrap official SkillRouter retrieval outputs, or run the official retrieval exporter when the released encoder is available:

```bash
python scripts/run_skillrouter_official.py \
  --skillrouter-root /Users/txx1220/Projects/skillrouter/SkillRouter \
  --tiers easy hard \
  --run-export \
  --encoder-model-or-path pipizhao/SkillRouter-Embedding-0.6B \
  --output-dir outputs/experiments_v2/skillrouter/official
```

By default the baseline uses the `raganything` backend. Install the RAG-Anything/LightRAG packages before running it:

```bash
pip install raganything lightrag-hku

python -m toollery.cli skill-rag-baseline \
  --scaled-data skill_scaled_data.jsonl \
  --out skill_scaled_raganything_predictions.jsonl \
  --rag-working-dir .raganything_skills \
  --top-k 3
```

For quick local smoke tests without external RAG dependencies, pass `--backend tfidf` explicitly.

## Use a Real LLM

```python
from toollery.llm import OpenAICompatibleLLM
from toollery.manual import synthesize_tool_manual
from toollery.pipeline import ToolleryAgent

llm = OpenAICompatibleLLM(model="gpt-5-mini")
manual = synthesize_tool_manual(tools, teacher=llm, verifier=llm)
agent = ToolleryAgent(tools, manual, selector=llm, tool_top_k=5)
call, candidates = agent.run("Book me a flight to Singapore tomorrow")
```

Required environment variables:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5-mini"
```

## Input Format

Tool files use this format:

```json
{
  "tools": [
    {
      "name": "get_weather_forecast",
      "description": "get the weather forecast for a city and date",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "date": {"type": "string"}
        }
      }
    }
  ]
}
```

Evaluation cases use this format:

```json
{
  "cases": [
    {
      "query": "Should I pack an umbrella tomorrow?",
      "ground_truth_tool": "get_weather_forecast"
    }
  ]
}
```
