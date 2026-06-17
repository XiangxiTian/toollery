import json
import http.client
import tempfile
import unittest
from pathlib import Path

from scripts.convert_car_tools_to_bfcl import (
    build_car_tools,
    load_bfcl_tools_file,
    write_tools_json,
)
from scripts.generate_car_tool_proxy_queries import (
    build_or_load_car_manual,
    generate_car_proxy_queries,
    generate_car_proxy_queries_batch,
    load_car_tool_specs,
    RetryingLLM,
    write_generated_queries,
)
import scripts.generate_car_tool_proxy_queries as car_proxy
from toollery.schemas import ManualEntry
from toollery.schemas import ToolSpec


class CarToolsConversionTest(unittest.TestCase):
    def test_groups_tool_rows_and_parameter_continuation_rows(self) -> None:
        rows = [
            {
                "actionNameZh": "打开车辆设备",
                "actionNameEn": "TurnOnCarDevice",
                "actionDescribe": "打开指定车辆设备",
                "inputNameZh": "设备类型",
                "inputNameEn": "deviceType",
                "dataType": "string",
                "inputDescribe": "用户指定的设备名称",
            },
            {
                "actionNameZh": "",
                "actionNameEn": "",
                "actionDescribe": "",
                "inputNameZh": "范围类型",
                "inputNameEn": "rangeType",
                "dataType": "string",
                "inputDescribe": "用户指定的操控范围",
            },
            {
                "actionNameZh": "车祸上报",
                "actionNameEn": "CarAccident",
                "actionDescribe": "报告车辆发生车祸事故",
                "inputNameZh": None,
                "inputNameEn": None,
                "dataType": None,
                "inputDescribe": None,
            },
        ]

        tools = build_car_tools(rows)

        self.assertEqual([tool["name"] for tool in tools], ["TurnOnCarDevice", "CarAccident"])
        self.assertEqual(tools[0]["description"], "打开指定车辆设备")
        self.assertEqual(tools[0]["parameters"]["type"], "dict")
        self.assertEqual(
            list(tools[0]["parameters"]["properties"]),
            ["deviceType", "rangeType"],
        )
        self.assertEqual(
            tools[0]["parameters"]["properties"]["deviceType"],
            {"type": "string", "description": "用户指定的设备名称"},
        )
        self.assertEqual(tools[0]["parameters"]["required"], [])
        self.assertEqual(tools[1]["parameters"], {"type": "dict", "properties": {}, "required": []})

    def test_preserves_round_trip_json_shape_for_bfcl_generation(self) -> None:
        tools = [
            {
                "name": "TurnOffCarDevice",
                "description": "关闭指定车辆设备",
                "parameters": {
                    "type": "dict",
                    "properties": {"deviceType": {"type": "string", "description": "设备名称"}},
                    "required": [],
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "car_tools.json"
            write_tools_json(path, tools)

            loaded = load_bfcl_tools_file(path)
            saved = json.loads(path.read_text(encoding="utf-8"))["tools"]

        self.assertEqual(loaded, tools)
        self.assertEqual(saved, tools)

    def test_car_proxy_query_script_loads_tools_json_without_bfcl_samples(self) -> None:
        tools = [
            {
                "name": "SetCarTemperature",
                "description": "设置车内温度",
                "parameters": {
                    "type": "dict",
                    "properties": {"temperature": {"type": "integer", "description": "目标温度"}},
                    "required": [],
                },
                "category": "vehicle_control",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "car_tools_bfcl.json"
            write_tools_json(path, tools)

            specs = load_car_tool_specs(path)

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "SetCarTemperature")
        self.assertEqual(specs[0].parameters["properties"]["temperature"]["type"], "integer")

    def test_car_proxy_query_script_writes_normalized_query_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.jsonl"

            write_generated_queries(
                path,
                [ManualEntry(tool_name="SetCarTemperature", query="把空调调到 24 度")],
            )

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows, [{"tool_name": "SetCarTemperature", "query": "把空调调到 24 度"}])

    def test_car_proxy_prompt_guides_implicit_vehicle_requests(self) -> None:
        class FakeLLM:
            prompt = ""

            def _chat(self, prompt: str, **_: object) -> str:
                self.prompt = prompt
                return '[{"query":"车里好热，帮我凉快一点","scenario_type":"implicit_comfort_request","generation_notes":"hot cabin implies cooling control"}]'

        llm = FakeLLM()
        tool = ToolSpec(
            name="TurnOnCarDevice",
            description="打开车辆设备，可操作空调、车窗等设备。",
            parameters={"type": "dict", "properties": {}},
        )

        rows = generate_car_proxy_queries(llm, tool, count=1)

        self.assertEqual(rows[0]["query"], "车里好热，帮我凉快一点")
        self.assertIn("implicit", llm.prompt.lower())
        self.assertIn("车里好热", llm.prompt)
        self.assertIn("空调", llm.prompt)
        self.assertIn("车窗", llm.prompt)

    def test_car_proxy_batch_prompt_returns_queries_by_exact_tool_name(self) -> None:
        class FakeLLM:
            prompt = ""

            def _chat(self, prompt: str, **_: object) -> str:
                self.prompt = prompt
                return '{"Vehicle.Cabin.HVAC.Row1.Left.Temperature":[{"query":"车里现在多少度","scenario_type":"implicit_status_query"}]}'

        llm = FakeLLM()
        tool = ToolSpec(
            name="Vehicle.Cabin.HVAC.Row1.Left.Temperature",
            description="查询空调模块的空调温度。",
            parameters={"type": "dict", "properties": {}},
        )

        rows_by_tool = generate_car_proxy_queries_batch(llm, [tool], count=1)

        self.assertEqual(rows_by_tool[tool.name][0]["query"], "车里现在多少度")
        self.assertIn("exact tool names", llm.prompt)
        self.assertIn(tool.name, llm.prompt)

    def test_retrying_llm_retries_remote_disconnects(self) -> None:
        class FlakyLLM:
            calls = 0

            def _chat(self, prompt: str, **_: object) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise http.client.RemoteDisconnected("closed")
                return '[{"query":"打开车窗透透气","scenario_type":"implicit_comfort_request"}]'

        inner = FlakyLLM()
        llm = RetryingLLM(inner, max_retries=1, retry_sleep=0)
        tool = ToolSpec(
            name="TurnOnCarDevice",
            description="打开车辆设备",
            parameters={"type": "dict", "properties": {}},
        )

        rows = generate_car_proxy_queries(llm, tool, count=1)

        self.assertEqual(inner.calls, 2)
        self.assertEqual(rows[0]["query"], "打开车窗透透气")

    def test_build_manual_refills_until_accepted_count_reaches_target_with_verifier(self) -> None:
        class FakeLLM:
            def _chat(self, prompt: str, **_: object) -> str:
                return '[{"query":"车里还是有点闷，继续开窗","scenario_type":"implicit_comfort_request"}]'

        original_verify = car_proxy.verify_proxy_query
        car_proxy.verify_proxy_query = lambda llm, query, tools: "TurnOnCarDevice"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw = root / "manual.raw.jsonl"
                manual_path = root / "manual.json"
                raw.write_text(
                    '{"stage":"candidate","skill_id":"TurnOnCarDevice","query_index":0,'
                    '"query":"打开车窗透气","accepted":true}\n'
                    '{"stage":"candidate","skill_id":"TurnOnCarDevice","query_index":1,'
                    '"query":"把空调温度设低","accepted":false,'
                    '"rejection_reason":"verifier_selected_different_skill"}\n',
                    encoding="utf-8",
                )

                rows, manual = build_or_load_car_manual(
                    manual_raw_path=raw,
                    manual_path=manual_path,
                    selected_skill_ids=["TurnOnCarDevice"],
                    pool_by_id={
                        "TurnOnCarDevice": ToolSpec(
                            name="TurnOnCarDevice",
                            description="打开车辆设备",
                            parameters={"type": "dict", "properties": {}},
                        )
                    },
                    example_queries={},
                    llm=FakeLLM(),
                    proxy_queries_per_skill=2,
                    verifier_distractors=0,
                    verify_proxies=True,
                    force_rebuild=False,
                    seed=31,
                    llm_workers=1,
                    llm_batch_size=1,
                )
        finally:
            car_proxy.verify_proxy_query = original_verify

        self.assertEqual(len(rows), 3)
        self.assertEqual([entry.query for entry in manual], ["打开车窗透气", "车里还是有点闷，继续开窗"])


if __name__ == "__main__":
    unittest.main()
