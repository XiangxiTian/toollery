# Car Proxy Recall Experiment Package

This package contains the scripts needed to run car-control proxy-query recall experiments.

It is a copy of the experiment code from `toollery_emnlp`, arranged so it can be run from the package root. It does not include any real API key.

## Files

```text
car_proxy_recall_experiment_package/
  README.md
  configs/
    car_proxy_recall_dashscope.full.example.json
  data/
    car_multintent_samples.jsonl
    car_tools_proxy_queries.jsonl
  scripts/
    run_car_proxy_recall.py
    run_full_dashscope.sh
  toollery/
    __init__.py
    embeddings.py
    text.py
  tests/
    test_car_proxy_recall.py
  examples/
    subset_predictions.jsonl
    subset_summary.json
```

## Inputs

The recall script expects two JSONL inputs.

Sample file format:

```json
{"sample_id": "car_00001", "query": "打开头枕音箱的导航音量和上锁右边的儿童车门", "correct_tools": ["TurnOffCarDeviceQuietMode", "LockCarDevice"]}
```

`original_query` is optional. If present, it is copied into the prediction output:

```json
{"sample_id": "car_00001", "query": "打开头枕音箱的导航音量和上锁右边的儿童车门", "original_query": "打开头枕音箱导航音量,上锁右边儿童车门锁", "correct_tools": ["TurnOffCarDeviceQuietMode", "LockCarDevice"]}
```

Proxy-query file format:

```json
{"tool_name": "TurnUpCarDeviceScreenSize", "query": "把投影幕布放大一点，字太小了看不清"}
```

Default input paths in the config point to the data files already included in this package:

```text
data/car_multintent_samples.jsonl
data/car_tools_proxy_queries.jsonl
```

So after unpacking the package, you do not need to copy additional data files for the default full experiment.

## Methods

`run_car_proxy_recall.py` supports:

- `bm25`: local BM25 retrieval over proxy queries.
- `llm`: embedding-model recall. The name is kept for command compatibility, but this method does not call chat completions. It encodes all proxy queries with the configured embedding API, caches the vectors, encodes each incoming query with the same embedding model, and returns top-k tools by vector similarity aggregated at tool level.
- `embedding`: alias with the same behavior as `llm`.

## DashScope Config

For DashScope OpenAI-compatible embeddings, use:

```json
"embedding": {
  "api_key": "PASTE_YOUR_API_KEY_HERE",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "text-embedding-v4",
  "batch_size": 10,
  "timeout": 120,
  "max_retries": 1
}
```

Important notes from the smoke test:

- `https://dashscope.aliyuncs.com` returned `404 Not Found` for `/embeddings`.
- `https://dashscope.aliyuncs.com/compatible-mode/v1` reached the embeddings API.
- `text-embedding-v4` rejected batch sizes larger than 10, so keep `batch_size` at `10` or lower.

## Run Full Experiment

Copy the example config and put your real key in the copy:

```bash
cp configs/car_proxy_recall_dashscope.full.example.json configs/car_proxy_recall_dashscope.full.json
```

Edit:

```json
"api_key": "PASTE_YOUR_API_KEY_HERE"
```

Then run:

```bash
python scripts/run_car_proxy_recall.py \
  --config configs/car_proxy_recall_dashscope.full.json
```

or:

```bash
bash scripts/run_full_dashscope.sh configs/car_proxy_recall_dashscope.full.json
```

The first full embedding run can take a while because it encodes all proxy queries. It writes reusable proxy-query embeddings to:

```text
outputs/car_proxy_recall_dashscope/proxy_query_embeddings.jsonl
```

Later runs reuse that cache if the proxy-query file and embedding config are unchanged.

To force rebuilding the embedding cache:

```bash
python scripts/run_car_proxy_recall.py \
  --config configs/car_proxy_recall_dashscope.full.json \
  --force-rebuild-embeddings
```

## Outputs

Prediction JSONL path from the example config:

```text
outputs/car_proxy_recall_dashscope/full_predictions.jsonl
```

Each row contains:

```json
{
  "method_name": "llm",
  "sample_id": "car_00000",
  "query": "...",
  "correct_tools": ["..."],
  "retrieved_tools": ["..."],
  "retrieved_candidates": [
    {
      "tool_name": "...",
      "score": 0.123,
      "supporting_queries": ["..."]
    }
  ],
  "top_k_hit": true,
  "final_prediction": "...",
  "final_success": false,
  "latency_ms": 12.3
}
```

Summary JSON path from the example config:

```text
outputs/car_proxy_recall_dashscope/full_summary.json
```

It reports per method:

- `samples`
- `labeled_samples`
- `top_k`
- `recall_at_k`
- `top_1_accuracy`
- `avg_latency_ms`

## Smoke-Test Result Included

The `examples/` directory contains the successful DashScope small-subset smoke test result:

```text
examples/subset_predictions.jsonl
examples/subset_summary.json
```

That smoke test used only the first 10 proxy queries and 1 sample. It verifies API connectivity and cache flow, not retrieval quality.

The result was:

```json
{
  "llm": {
    "samples": 1,
    "labeled_samples": 1,
    "top_k": 5,
    "recall_at_k": 0.0,
    "top_1_accuracy": 0.0,
    "avg_latency_ms": 1814.825082954485
  }
}
```

## Run Tests

From the package root:

```bash
python -m unittest tests.test_car_proxy_recall
python -m compileall scripts/run_car_proxy_recall.py
```
