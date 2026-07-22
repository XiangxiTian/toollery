import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_car_proxy_recall import (
    BM25ProxyRetriever,
    CarRecallSample,
    EmbeddingProxyRetriever,
    apply_config,
    make_embedder,
    load_proxy_queries,
    load_samples,
    parse_args,
    run_recall,
)


class CarProxyRecallTest(unittest.TestCase):
    def test_loads_multintent_samples_and_proxy_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "samples.jsonl"
            proxies_path = root / "proxies.jsonl"
            samples_path.write_text(
                json.dumps(
                    {
                        "sample_id": "car_00000",
                        "query": "车里好热，把窗户打开",
                        "correct_tools": ["OpenWindow"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            proxies_path.write_text(
                json.dumps({"tool_name": "OpenWindow", "query": "打开车窗透透气"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            samples = load_samples(samples_path)
            proxies = load_proxy_queries(proxies_path)

        self.assertEqual(
            samples,
            [CarRecallSample(sample_id="car_00000", query="车里好热，把窗户打开", correct_tools=["OpenWindow"])],
        )
        self.assertEqual(proxies, {"OpenWindow": ["打开车窗透透气"]})

    def test_preserves_optional_original_query_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples_path = Path(tmp) / "samples.jsonl"
            samples_path.write_text(
                json.dumps(
                    {
                        "sample_id": "car_00001",
                        "query": "打开头枕音箱的导航音量和上锁右边的儿童车门",
                        "original_query": "打开头枕音箱导航音量,上锁右边儿童车门锁",
                        "correct_tools": ["TurnOffCarDeviceQuietMode", "LockCarDevice"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            samples = load_samples(samples_path)

        self.assertEqual(samples[0].original_query, "打开头枕音箱导航音量,上锁右边儿童车门锁")

    def test_bm25_retrieves_top_k_tools_from_proxy_queries(self) -> None:
        retriever = BM25ProxyRetriever(
            {
                "OpenWindow": ["打开车窗透透气", "车里闷，帮我开窗"],
                "SetTemperature": ["空调调到二十四度"],
                "LockDoor": ["锁上车门", "把儿童锁打开"],
            }
        )

        hits = retriever.retrieve("帮我打开车窗通风", top_k=2, proxy_top_k=10)

        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["tool_name"], "OpenWindow")
        self.assertGreater(hits[0]["score"], hits[1]["score"])

    def test_embedding_retriever_caches_proxy_vectors_and_recalls_with_query_embedding(self) -> None:
        class FakeEmbedder:
            def __init__(self, fail_on_many: bool = False) -> None:
                self.fail_on_many = fail_on_many
                self.encode_many_calls = 0
                self.encode_calls = 0

            def fit(self, texts: list[str]) -> "FakeEmbedder":
                return self

            def encode_many(self, texts: list[str]) -> list[list[float]]:
                self.encode_many_calls += 1
                if self.fail_on_many:
                    raise AssertionError("cached proxy vectors should be reused")
                return [self._vector(text) for text in texts]

            def encode(self, text: str) -> list[float]:
                self.encode_calls += 1
                return self._vector(text)

            @staticmethod
            def _vector(text: str) -> list[float]:
                if "窗" in text or "通风" in text:
                    return [1.0, 0.0]
                if "锁" in text:
                    return [0.0, 1.0]
                return [0.0, 0.0]

        proxies = {
            "OpenWindow": ["打开车窗透透气"],
            "LockDoor": ["锁车门"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "proxy_vectors.jsonl"
            first_embedder = FakeEmbedder()
            first = EmbeddingProxyRetriever(proxies, embedder=first_embedder, cache_path=cache_path)
            first_hits = first.retrieve("帮我开窗通风", top_k=1)

            second_embedder = FakeEmbedder(fail_on_many=True)
            second = EmbeddingProxyRetriever(proxies, embedder=second_embedder, cache_path=cache_path)
            second_hits = second.retrieve("帮我开窗通风", top_k=1)

        self.assertEqual(first_embedder.encode_many_calls, 1)
        self.assertEqual([hit["tool_name"] for hit in first_hits], ["OpenWindow"])
        self.assertEqual(second_embedder.encode_many_calls, 0)
        self.assertEqual(second_embedder.encode_calls, 1)
        self.assertEqual([hit["tool_name"] for hit in second_hits], ["OpenWindow"])

    def test_run_recall_writes_prediction_rows_and_summary(self) -> None:
        samples = [
            CarRecallSample(sample_id="car_0", query="打开车窗", correct_tools=["OpenWindow"]),
            CarRecallSample(sample_id="car_1", query="锁车", correct_tools=["LockDoor"]),
        ]
        proxies = {
            "OpenWindow": ["打开车窗"],
            "LockDoor": ["锁车门"],
        }

        rows, summary = run_recall(
            samples=samples,
            proxy_queries_by_tool=proxies,
            methods=["bm25"],
            top_k=1,
            bm25_proxy_top_k=5,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["method_name"], "bm25")
        self.assertEqual(rows[0]["retrieved_tools"], ["OpenWindow"])
        self.assertEqual(summary["bm25"]["samples"], 2)
        self.assertEqual(summary["bm25"]["recall_at_k"], 1.0)

    def test_run_recall_llm_method_uses_embedding_retrieval(self) -> None:
        class FakeEmbedder:
            def fit(self, texts: list[str]) -> "FakeEmbedder":
                return self

            def encode_many(self, texts: list[str]) -> list[list[float]]:
                return [self.encode(text) for text in texts]

            def encode(self, text: str) -> list[float]:
                return [1.0, 0.0] if "窗" in text else [0.0, 1.0]

        with tempfile.TemporaryDirectory() as tmp:
            rows, summary = run_recall(
                samples=[CarRecallSample(sample_id="car_0", query="打开车窗", correct_tools=["OpenWindow"])],
                proxy_queries_by_tool={"OpenWindow": ["打开车窗"], "LockDoor": ["锁车门"]},
                methods=["llm"],
                top_k=1,
                bm25_proxy_top_k=5,
                embedder=FakeEmbedder(),
                embedding_cache=Path(tmp) / "proxy_vectors.jsonl",
            )

        self.assertEqual(rows[0]["method_name"], "llm")
        self.assertEqual(rows[0]["retrieved_tools"], ["OpenWindow"])
        self.assertEqual(summary["llm"]["recall_at_k"], 1.0)

    def test_apply_config_respects_explicit_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "car_proxy_recall": {
                            "methods": ["bm25", "llm"],
                            "top_k": 20,
                            "samples": "from_config.jsonl",
                            "embedding_cache": "cache/from_config.jsonl",
                        },
                        "embedding": {
                            "api_key": "test-embedding-key",
                            "base_url": "https://embedding.example/v1",
                            "model": "embedding-model",
                        }
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(["--config", str(config_path), "--top-k", "3"])
            apply_config(args)

        self.assertEqual(args.methods, ["bm25", "llm"])
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.samples, "from_config.jsonl")
        self.assertEqual(args.embedding_cache, "cache/from_config.jsonl")
        self.assertEqual(args.embedding_config["api_key"], "test-embedding-key")

    def test_make_embedder_reads_embedding_api_key_directly_from_config(self) -> None:
        embedder = make_embedder(
            {
                "api_key": "test-embedding-key",
                "base_url": "https://embedding.example/v1",
                "model": "embedding-model",
                "dimensions": 1024,
                "batch_size": 16,
                "timeout": 7,
            }
        )

        self.assertEqual(embedder.api_key, "test-embedding-key")
        self.assertEqual(embedder.base_url, "https://embedding.example/v1")
        self.assertEqual(embedder.model, "embedding-model")
        self.assertEqual(embedder.dimensions, 1024)
        self.assertEqual(embedder.batch_size, 16)
        self.assertEqual(embedder.timeout, 7)


if __name__ == "__main__":
    unittest.main()
