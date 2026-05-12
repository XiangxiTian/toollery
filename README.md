# Toollery Python Flow

这是根据论文 `Toollery: Scaling LLM Agents to Thousands of Tools` 实现的端到端 Python 代码流程。核心思想是把大规模工具选择拆成两步：

1. 离线阶段：为每个工具生成多样化 proxy queries，并通过 round-trip verification 过滤掉不稳定样本。
2. 在线阶段：用户请求先和 proxy queries 做 intent-to-intent 检索，再把聚合后的 top-k 工具交给最终选择器。

代码默认使用无外部服务的启发式 teacher/verifier/selector，方便直接跑通。生产环境可以把 `OpenAICompatibleLLM` 作为 teacher、verifier 或最终 selector。

## 目录

- `toollery/manual.py`：论文 Algorithm 1，对工具元数据合成并验证 proxy queries。
- `toollery/retrieval.py`：proxy-query 检索、按工具聚合分数、构造 compact candidate set。
- `toollery/pipeline.py`：在线推理流程，先召回候选工具，再生成工具调用。
- `toollery/scaletool.py`：ScaleTool 风格的候选集增长评测。
- `toollery/cli.py`：命令行入口。
- `examples/`：一组可运行的工具和测试样例。

## 快速运行

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

## 使用真实 LLM

```python
from toollery.llm import OpenAICompatibleLLM
from toollery.manual import synthesize_tool_manual
from toollery.pipeline import ToolleryAgent

llm = OpenAICompatibleLLM(model="gpt-5-mini")
manual = synthesize_tool_manual(tools, teacher=llm, verifier=llm)
agent = ToolleryAgent(tools, manual, selector=llm, tool_top_k=5)
call, candidates = agent.run("Book me a flight to Singapore tomorrow")
```

需要设置：

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5-mini"
```

## 输入格式

工具文件支持：

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

评测样例支持：

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
