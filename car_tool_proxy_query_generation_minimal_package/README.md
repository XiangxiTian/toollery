# Car Tool Proxy Query Generation Minimal Package

This package contains the minimal runnable chain for:

```bash
python3 scripts/generate_car_tool_proxy_queries.py
```

No source code files were changed. The Python files here are copied as-is from the source repository.

## Contents

- `scripts/generate_car_tool_proxy_queries.py`: entry script
- `scripts/convert_car_tools_to_bfcl.py`: tool JSON loader used by the entry script
- `scripts/run_skillrouter_toollery.py`: shared config, progress, JSON extraction, and verifier helpers
- `toollery/`: minimal imported runtime modules
- `outputs/car_tools/car_tools_bfcl.json`: default vehicle tool metadata input
- `skillrouter_toollery_deepseek_config.example.json`: sanitized default config loaded by the script
- `requirements.txt`: no required third-party Python packages for this path

## Requirements

- Python 3.10+
- Network access to an OpenAI-compatible chat-completions endpoint
- An API key exported as `LLM_API_KEY`

The default config points to DeepSeek-compatible chat completions. To use another provider, edit only the config file or pass `--config` with your own JSON file.

## Run

From this package directory:

```bash
export LLM_API_KEY="YOUR_API_KEY"
python3 scripts/generate_car_tool_proxy_queries.py \
  --out outputs/car_tools/car_tools_proxy_queries.jsonl \
  --summary-out outputs/car_tools/car_tools_proxy_queries.summary.json
```

For a small smoke run:

```bash
export LLM_API_KEY="YOUR_API_KEY"
python3 scripts/generate_car_tool_proxy_queries.py \
  --out outputs/car_tools/car_tools_proxy_queries.sample.jsonl \
  --summary-out outputs/car_tools/car_tools_proxy_queries.sample.summary.json \
  --limit-tools 2 \
  --proxy-queries-per-tool 1 \
  --llm-workers 1 \
  --llm-batch-size 1 \
  --force-rebuild
```

The script also writes resumable intermediate files beside `--out` unless overridden:

- `*.raw.jsonl`
- `*.manual.json`

