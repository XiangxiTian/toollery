import tempfile
import unittest
from pathlib import Path

from scripts.run_insight_tool_predictions import (
    InsightQuery,
    load_insight_tools,
    load_unlabeled_queries,
    prediction_rows,
)
from scripts.run_taichu_toollery_bm25 import TaichuPair, make_taichu_tools


class InsightToolPredictionsTest(unittest.TestCase):
    def test_loads_insight_tools_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tools.json"
            path.write_text(
                '{"tools":[{"name":"playMusic","description":"播放音乐","parameters":{"type":"object"}}]}',
                encoding="utf-8",
            )

            tools = load_insight_tools(path)

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "playMusic")
        self.assertEqual(tools[0].description, "播放音乐")
        self.assertEqual(tools[0].category, "insight_tool")

    def test_loads_queries_without_using_gold_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.json"
            path.write_text(
                '[{"query":"播放音乐","expect_intent":"0.播放控制-关闭音乐"}]',
                encoding="utf-8",
            )

            queries = load_unlabeled_queries(path)

        self.assertEqual(queries, [InsightQuery(sample_id="query_00000", query="播放音乐")])

    def test_prediction_rows_do_not_require_truth(self) -> None:
        tools = make_taichu_tools(["playMusic", "stopMusic"])
        proxy_pairs = [
            TaichuPair(sample_id="proxy_0", query="放首歌", expect_intent="playMusic"),
            TaichuPair(sample_id="proxy_1", query="停止播放", expect_intent="stopMusic"),
        ]
        queries = [InsightQuery(sample_id="q0", query="播放音乐")]

        rows = prediction_rows(
            tools=tools,
            proxy_pairs=proxy_pairs,
            queries=queries,
            retriever_name="bm25",
            top_k=2,
        )

        self.assertEqual(rows[0]["sample_id"], "q0")
        self.assertEqual(rows[0]["query"], "播放音乐")
        self.assertIn("predicted_tools", rows[0])
        self.assertNotIn("correct_intent", rows[0])


if __name__ == "__main__":
    unittest.main()
