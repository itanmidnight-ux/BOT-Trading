#!/usr/bin/env bash
# start_training.sh -- Entrenamiento offline Linux (sin MT5)
set -uo pipefail
cd "$(dirname "$0")"

echo "===================================================="
echo "  BOT-Trading v4.1 -- Modo Entrenamiento (sin MT5)"
echo "  Usa yfinance para datos historicos EUR/USD"
echo "===================================================="
echo

# shellcheck disable=SC1091
source .venv/bin/activate

python -c "
import sys
from unittest.mock import MagicMock
sys.modules['MetaTrader5'] = MagicMock()

from config import settings
from core.feature_engine import FeatureEngine
from core.model_trainer import ModelTrainer
from core.training_loop import TrainingLoop
from core.model_evaluator import ModelEvaluator
from core.backtester import Backtester
from core.regime_detector import RegimeDetector

import yfinance as yf
import pandas as pd

print('[1/4] Descargando datos EUR/USD...')
df = yf.download('EURUSD=X', period='90d', interval='1h', progress=False, auto_adjust=True)
df = df.reset_index()
df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
df = df.rename(columns={'datetime': 'time', 'volume': 'tick_volume'})
df['tick_volume'] = df.get('tick_volume', 500)
df['spread'] = 10
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']].dropna()
print(f'  OK: {len(df)} velas EUR/USD H1')

print('[2/4] Generando features...')
fe = FeatureEngine()
df_feat = fe.build(df)
print(f'  OK: {df_feat.shape[1]} features')

print('[3/4] Entrenando modelo (gate 59% WR)...')
trainer = ModelTrainer()
evaluator = ModelEvaluator()
regime = RegimeDetector()
loop = TrainingLoop('EURUSD', fe, trainer, evaluator, Backtester, regime)
result = loop.run(df_feat, initial_capital=settings.INITIAL_CAPITAL, verbose=True)
wr = result.get('win_rate', 0)
print(f'  WR={wr:.1%} | Ready={result.get(\"ready_for_live\", False)}')

print('[4/4] Guardando modelo...')
print()
print('Entrenamiento OK. Para live: configura MT5 (README) + python main.py')
"
