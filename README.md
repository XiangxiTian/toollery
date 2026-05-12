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
