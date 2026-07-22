#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/car_proxy_recall_dashscope.full.example.json}"

python scripts/run_car_proxy_recall.py --config "$CONFIG_PATH"
