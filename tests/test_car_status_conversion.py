import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_car_status_to_bfcl import build_status_tools, parse_getter_signatures, write_tools_json


class CarStatusConversionTest(unittest.TestCase):
    def test_uses_vspec_signal_as_tool_name_and_getter_args_as_parameters(self) -> None:
        rows = [
            {
                "所属模块": "车门",
                "信号名称": "车门开关状态",
                "北向API": "boolean getDoorOpenStatus(DoorZone zoneId)",
                "北向API接口枚举值或范围": "false=表示关闭\ntrue=表示打开",
                "vspec信号": "Vehicle.Cabin.Door.Row1.Left.IsOpen",
                "vspec类型": "BOOLEAN",
            }
        ]

        tools = build_status_tools(rows)

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "Vehicle.Cabin.Door.Row1.Left.IsOpen")
        self.assertEqual(tools[0]["category"], "vehicle_status_query")
        self.assertIn("查询车门模块的车门开关状态", tools[0]["description"])
        self.assertIn("false=表示关闭", tools[0]["description"])
        self.assertEqual(
            tools[0]["parameters"],
            {
                "type": "dict",
                "properties": {
                    "zoneId": {
                        "type": "string",
                        "description": "DoorZone zoneId，用于指定查询车门开关状态时的车辆区域或位置。",
                    }
                },
                "required": [],
            },
        )

    def test_merges_parameters_from_multiple_getters_for_one_signal_row(self) -> None:
        rows = [
            {
                "所属模块": "车门",
                "信号名称": "左前门故障原因",
                "北向API": (
                    "int getRearDoorFaultReason(),\n"
                    "int getSideDoorFaultReason(DoorZone zoneId)"
                ),
                "北向API接口枚举值或范围": "0=正常\n15=错误类型",
                "vspec信号": "Vehicle.Cabin.Door.Row1.Left.FaultReason",
                "vspec类型": "INT32",
            }
        ]

        tools = build_status_tools(rows)

        self.assertEqual(list(tools[0]["parameters"]["properties"]), ["zoneId"])
        self.assertIn("getRearDoorFaultReason", tools[0]["description"])
        self.assertIn("getSideDoorFaultReason", tools[0]["description"])

    def test_includes_rows_without_getter_by_default(self) -> None:
        rows = [
            {
                "所属模块": "车窗",
                "信号名称": "车窗升降",
                "北向API": "void setWindowLiftStatus(WindowZone zoneId, int setValue)",
                "北向API接口枚举值或范围": "0=无效",
                "vspec信号": "Vehicle.Cabin.Door.Row1.Left.Window.Switch",
                "vspec类型": "INT32",
            }
        ]

        tools = build_status_tools(rows)

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "Vehicle.Cabin.Door.Row1.Left.Window.Switch")
        self.assertEqual(tools[0]["parameters"], {"type": "dict", "properties": {}, "required": []})
        self.assertIn("车窗升降", tools[0]["description"])

    def test_can_keep_query_only_rows_when_requested(self) -> None:
        rows = [
            {
                "所属模块": "车窗",
                "信号名称": "车窗升降",
                "北向API": "void setWindowLiftStatus(WindowZone zoneId, int setValue)",
                "北向API接口枚举值或范围": "0=无效",
                "vspec信号": "Vehicle.Cabin.Door.Row1.Left.Window.Switch",
                "vspec类型": "INT32",
            }
        ]

        self.assertEqual(build_status_tools(rows, include_without_getter=False), [])

    def test_merges_duplicate_vspec_rows_instead_of_dropping_later_getters(self) -> None:
        rows = [
            {
                "所属模块": "车窗",
                "信号名称": "车窗升降",
                "北向API": "void setWindowLiftStatus(WindowZone zoneId, int setValue)",
                "北向API接口枚举值或范围": "0=无效",
                "vspec信号": "Vehicle.Cabin.Door.Row1.Left.Window.Switch",
                "vspec类型": "INT32",
            },
            {
                "所属模块": "车窗",
                "信号名称": "车窗升降查询",
                "北向API": "int getWindowLiftStatus(WindowZone zoneId)",
                "北向API接口枚举值或范围": "1=上升",
                "vspec信号": "Vehicle.Cabin.Door.Row1.Left.Window.Switch",
                "vspec类型": "INT32",
            },
        ]

        tools = build_status_tools(rows)

        self.assertEqual(len(tools), 1)
        self.assertIn("车窗升降", tools[0]["description"])
        self.assertIn("车窗升降查询", tools[0]["description"])
        self.assertEqual(list(tools[0]["parameters"]["properties"]), ["zoneId"])

    def test_parses_common_getter_signatures(self) -> None:
        signatures = parse_getter_signatures(
            "float getWindowPositionInfo(WindowZone zoneId),\n"
            "String getCustomPropertyWithZone(String propId, int zoneId)"
        )

        self.assertEqual([signature.name for signature in signatures], ["getWindowPositionInfo", "getCustomPropertyWithZone"])
        self.assertEqual([(param.type_name, param.name) for param in signatures[1].parameters], [("String", "propId"), ("int", "zoneId")])

    def test_writes_top_level_tools_json(self) -> None:
        tools = [
            {
                "name": "Vehicle.Body.Lights.IsLeftIndicatorOn",
                "description": "查询左转向灯亮灭状态",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status_tools.json"

            write_tools_json(path, tools)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved, {"tools": tools})


if __name__ == "__main__":
    unittest.main()
