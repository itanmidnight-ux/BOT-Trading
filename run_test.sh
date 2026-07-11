#!/usr/bin/env bash
# run_test.sh -- Modo test Linux (sin MT5)
set -uo pipefail
cd "$(dirname "$0")"

echo "===================================================="
echo "  BOT-Trading v4.1 -- MODO TEST (sin MT5)"
echo "  Backtest completo + LLM con datos yfinance"
echo "===================================================="
echo

# shellcheck disable=SC1091
source .venv/bin/activate
python main.py --test
