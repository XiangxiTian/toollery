import unittest

from scripts.run_taichu_toollery_bm25 import (
    QMetaDenseIndex,
    QMetaBM25Index,
    TaichuPair,
    generate_or_load_proxy_queries,
    generate_taichu_proxy_rows_batch,
    load_proxy_query_pairs,
    make_taichu_tools,
    retrieval_metrics,
    run_dense_predictions,
    run_leave_one_out,
)


class TaichuToolleryBM25Test(unittest.TestCase):
    def test_retrieves_from_query_plus_metadata_without_current_query(self) -> None:
        tools = make_taichu_tools(["0.打开应用", "0.关闭应用"])
        pairs = [
            TaichuPair(sample_id="0", query="打开微信", expect_intent="0.打开应用"),
            TaichuPair(sample_id="1", query="启动微信", expect_intent="0.打开应用"),
            TaichuPair(sample_id="2", query="关闭微信", expect_intent="0.关闭应用"),
        ]

        index = QMetaBM25Index(tools, pairs)
        retrieved = index.retrieve("打开微信", top_k=2, exclude_queries={"打开微信"})

        self.assertEqual(retrieved[0].tool.name, "0.打开应用")
        self.assertTrue(
            all(unit.intent_query != "打开微信" for _, unit in index.search("打开微信", exclude_queries={"打开微信"}))
        )

    def test_dense_retriever_aggregates_query_plus_metadata_units(self) -> None:
        class FakeEmbedder:
            def fit(self, texts: list[str]) -> "FakeEmbedder":
                return self

            def encode_many(self, texts: list[str], progress_callback: object | None = None) -> list[list[float]]:
                return [self.encode(text) for text in texts]

            def encode(self, text: str) -> list[float]:
                if "打开" in text or "启动" in text:
                    return [1.0, 0.0]
                if "关闭" in text or "退出" in text:
                    return [0.0, 1.0]
                return [0.0, 0.0]

        tools = make_taichu_tools(["0.打开应用", "0.关闭应用"])
        pairs = [
            TaichuPair(sample_id="0", query="启动微信", expect_intent="0.打开应用"),
            TaichuPair(sample_id="1", query="退出音乐", expect_intent="0.关闭应用"),
        ]

        index = QMetaDenseIndex(tools, pairs, embedder=FakeEmbedder())
        retrieved = index.retrieve("打开地图", top_k=2)

        self.assertEqual(retrieved[0].tool.name, "0.打开应用")

    def test_metrics_report_ranked_retrieval_quality(self) -> None:
        rows = [
            {"correct_intent": "a", "retrieved_candidates": ["a", "b", "c"]},
            {"correct_intent": "b", "retrieved_candidates": ["a", "b", "c"]},
            {"correct_intent": "z", "retrieved_candidates": ["a", "b", "c"]},
        ]

        metrics = retrieval_metrics(rows)

        self.assertAlmostEqual(metrics["Hit@1"], 1 / 3)
        self.assertAlmostEqual(metrics["Recall@3"], 2 / 3)
        self.assertAlmostEqual(metrics["MRR@10"], (1.0 + 0.5 + 0.0) / 3)
        self.assertEqual(metrics["count"], 3)

    def test_bm25_predictions_include_scores_aligned_with_candidates(self) -> None:
        tools = make_taichu_tools(["0.打开应用", "0.关闭应用"])
        pairs = [
            TaichuPair(sample_id="0", query="打开微信", expect_intent="0.打开应用"),
            TaichuPair(sample_id="1", query="关闭微信", expect_intent="0.关闭应用"),
        ]

        rows = run_leave_one_out(tools=tools, pairs=pairs, top_k=2)

        self.assertEqual(len(rows[0]["scores"]), len(rows[0]["retrieved_candidates"]))
        self.assertTrue(all(isinstance(score, float) for score in rows[0]["scores"]))

    def test_dense_predictions_include_scores_aligned_with_candidates(self) -> None:
        class FakeEmbedder:
            def fit(self, texts: list[str]) -> "FakeEmbedder":
                return self

            def encode_many(self, texts: list[str], progress_callback: object | None = None) -> list[list[float]]:
                return [self.encode(text) for text in texts]

            def encode(self, text: str) -> list[float]:
                if "打开" in text or "启动" in text:
                    return [1.0, 0.0]
                if "关闭" in text or "退出" in text:
                    return [0.0, 1.0]
                return [0.0, 0.0]

        tools = make_taichu_tools(["0.打开应用", "0.关闭应用"])
        pairs = [
            TaichuPair(sample_id="0", query="启动微信", expect_intent="0.打开应用"),
            TaichuPair(sample_id="1", query="退出音乐", expect_intent="0.关闭应用"),
        ]

        rows = run_dense_predictions(
            tools=tools,
            pairs=pairs,
            proxy_pairs=pairs,
            top_k=2,
            embedder=FakeEmbedder(),
        )

        self.assertEqual(len(rows[0]["scores"]), len(rows[0]["retrieved_candidates"]))
        self.assertTrue(all(isinstance(score, float) for score in rows[0]["scores"]))

    def test_generates_proxy_rows_from_llm_batch_without_dataset_examples(self) -> None:
        class FakeLLM:
            prompt = ""

            def _chat(self, prompt: str, **_: object) -> str:
                self.prompt = prompt
                return """
                {
                  "0.打开应用": [
                    {"query": "帮我启动微信", "scenario_type": "app_control"},
                    {"query": "打开地图应用", "scenario_type": "app_control"}
                  ],
                  "0.关闭应用": [
                    {"query": "把当前软件关掉", "scenario_type": "app_control"},
                    {"query": "退出音乐应用", "scenario_type": "app_control"}
                  ]
                }
                """

        llm = FakeLLM()
        tools = make_taichu_tools(["0.打开应用", "0.关闭应用"])

        rows = generate_taichu_proxy_rows_batch(llm, tools, count=2)

        self.assertEqual([row["query"] for row in rows], ["帮我启动微信", "打开地图应用", "把当前软件关掉", "退出音乐应用"])
        self.assertTrue(all(row["accepted"] is True for row in rows))
        self.assertNotIn("query_intent_pair", llm.prompt)
        self.assertNotIn("expect_intent", llm.prompt)

    def test_loads_normalized_proxy_query_jsonl(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.jsonl"
            path.write_text(
                '{"tool_name":"0.打开应用","query":"打开微信"}\n'
                '{"intent":"0.关闭应用","query":"关闭音乐"}\n',
                encoding="utf-8",
            )

            pairs = load_proxy_query_pairs(path)

        self.assertEqual(
            pairs,
            [
                TaichuPair(sample_id="proxy_00000", query="打开微信", expect_intent="0.打开应用"),
                TaichuPair(sample_id="proxy_00001", query="关闭音乐", expect_intent="0.关闭应用"),
            ],
        )

    def test_proxy_generation_resumes_partial_raw_rows(self) -> None:
        import tempfile
        from pathlib import Path

        class FakeLLM:
            def _chat(self, prompt: str, **_: object) -> str:
                return '{"0.关闭应用":["关闭音乐","退出应用"]}'

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "manual_raw.jsonl"
            out = root / "generated.jsonl"
            raw.write_text(
                '{"stage":"candidate","skill_id":"0.打开应用","tool_name":"0.打开应用","query_index":0,'
                '"query":"打开微信","accepted":true}\n'
                '{"stage":"candidate","skill_id":"0.打开应用","tool_name":"0.打开应用","query_index":1,'
                '"query":"启动地图","accepted":true}\n',
                encoding="utf-8",
            )

            pairs = generate_or_load_proxy_queries(
                tools=make_taichu_tools(["0.打开应用", "0.关闭应用"]),
                output_path=out,
                manual_raw_path=raw,
                count=2,
                batch_size=2,
                llm=FakeLLM(),
                force_rebuild=False,
            )

        self.assertEqual(len(pairs), 4)
        self.assertEqual([pair.expect_intent for pair in pairs].count("0.打开应用"), 2)
        self.assertEqual([pair.expect_intent for pair in pairs].count("0.关闭应用"), 2)


if __name__ == "__main__":
    unittest.main()
