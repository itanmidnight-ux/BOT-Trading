#!/usr/bin/env bash
# run.sh — configura y lanza el bot en Linux.
#
# Uso:
#   ./run.sh                 lanza el bot (DRY_RUN segun .env; default True = sin ordenes reales)
#   ./run.sh --test          corre la suite de tests y sale
#   ./run.sh --backtest CSV  corre el backtester sobre un CSV de velas M1 propias
#   ./run.sh --check         valida entorno/config sin lanzar el bot
set -euo pipefail
cd "$(dirname "$0")"

BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}[run]${NC} $*"; }
ok()    { echo -e "${GREEN}[run]${NC} $*"; }
warn()  { echo -e "${YELLOW}[run]${NC} $*"; }
fail()  { echo -e "${RED}[run]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Instalacion automatica si falta el entorno
# ---------------------------------------------------------------------------
if [ ! -d .venv ]; then
    info "No hay entorno instalado; ejecutando ./install.sh primero"
    ./install.sh
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 2. Configuracion privada (.env)
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    warn "Se creo .env desde la plantilla. Edita tus credenciales MT5 en .env"
fi
chmod 600 .env
mkdir -p logs

# ---------------------------------------------------------------------------
# 3. Modos de ejecucion
# ---------------------------------------------------------------------------
MODE="${1:-run}"

case "$MODE" in
    --test)
        info "Corriendo suite de tests..."
        exec python -m pytest tests/ -v
        ;;

    --backtest)
        CSV="${2:-}"
        [ -n "$CSV" ] || fail "Uso: ./run.sh --backtest ruta/a/velas_m1.csv"
        [ -f "$CSV" ] || fail "No existe el archivo: $CSV"
        info "Corriendo backtest sobre $CSV"
        exec python backtester.py "$CSV"
        ;;

    --check)
        info "Validando entorno y configuracion..."
        python - <<'EOF'
import config
from mt5_compat import BACKEND

print(f"  Simbolo:           {config.SYMBOL} @ {config.TIMEFRAME}")
print(f"  DRY_RUN:           {config.DRY_RUN}")
print(f"  Backend MT5:       {BACKEND or 'NO DISPONIBLE (instala MetaTrader5 o levanta mt5linux)'}")
print(f"  Riesgo por trade:  {config.RISK_PER_TRADE_PCT}% (techo {config.MAX_RISK_PER_TRADE_PCT}%)")
print(f"  Perdida diaria max:{config.MAX_DAILY_LOSS_PCT}%  Drawdown max: {config.MAX_DRAWDOWN_PCT}%")
print(f"  Grid:              {'ON' if config.GRID_ENABLED else 'OFF'} ({config.GRID_LEVELS} niveles, tope {config.GRID_MAX_TOTAL_RISK_PCT}%)")
print(f"  Credenciales MT5:  {'configuradas' if config.MT5_LOGIN else 'NO configuradas (usara sesion ya logueada del terminal)'}")
EOF
        ok "Configuracion valida."
        exit 0
        ;;

    run)
        # Aviso de seguridad segun DRY_RUN actual
        DRY=$(python -c "import config; print(config.DRY_RUN)")
        if [ "$DRY" = "True" ]; then
            info "DRY_RUN=True: el bot calcula y loguea señales SIN enviar ordenes reales."
        else
            warn "DRY_RUN=False: el bot ENVIARA ORDENES REALES a la cuenta configurada."
            warn "Ctrl+C en los proximos 5 segundos para abortar..."
            sleep 5
        fi
        info "Lanzando bot (logs en logs/)..."
        exec python main.py
        ;;

    *)
        fail "Modo desconocido: $MODE (usa: sin argumentos, --test, --backtest CSV, --check)"
        ;;
esac
