#!/usr/bin/env bash
# install.sh -- Instalador Linux para BOT-Trading
# MT5/FBS | Ensemble ML + Ollama LLM | Kelly 1:3000
set -uo pipefail
cd "$(dirname "$0")"

echo
echo "===================================================="
echo "   BOT-Trading v4.1 -- Instalador Linux"
echo "   MT5/FBS | Ensemble ML + Ollama LLM | Kelly 1:3000"
echo "===================================================="
echo

# 1. Python
echo "[1/7] Verificando Python..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python no encontrado. Instala Python 3.8+ (ej: sudo apt install python3 python3-venv)"
    exit 1
fi
python3 -c "import sys; v=sys.version_info; exit(0 if v.major==3 and v.minor>=8 else 1)" || {
    echo "[ERROR] Requiere Python 3.8+"
    exit 1
}
echo "[OK] Python OK ($(python3 --version))"

# 2. Wine (necesario para correr MT5 -- solo tiene binarios Windows)
echo "[2/7] Verificando Wine (necesario para MetaTrader 5 en Linux)..."
if command -v wine >/dev/null 2>&1; then
    echo "[OK] Wine encontrado: $(wine --version)"
else
    echo "[WARN] Wine no encontrado."
    echo "  MT5 solo distribuye binarios Windows; en Linux se ejecuta vía Wine."
    echo "  Instala Wine, p.ej.: sudo apt install wine64 winetricks xvfb"
fi

# 3. Xvfb (display virtual para correr MT5 sin entorno gráfico)
echo "[3/7] Verificando Xvfb (display virtual para servidores headless)..."
if command -v Xvfb >/dev/null 2>&1; then
    echo "[OK] Xvfb encontrado"
else
    echo "[WARN] Xvfb no encontrado. Instálalo si el servidor no tiene entorno gráfico:"
    echo "  sudo apt install xvfb"
fi

WINEPREFIX_DEFAULT="$HOME/.wine_mt5"
MT5_DEFAULT="$WINEPREFIX_DEFAULT/drive_c/Program Files/MetaTrader 5/terminal64.exe"
if [ -f "$MT5_DEFAULT" ]; then
    echo "[OK] MT5 encontrado: $MT5_DEFAULT"
    MT5_EXE="$MT5_DEFAULT"
else
    echo "[WARN] MT5 no encontrado en $MT5_DEFAULT"
    echo "  Descarga el instalador Windows desde:"
    echo "  https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
    echo "  E instálalo dentro de Wine, ej:"
    echo "    WINEPREFIX=$WINEPREFIX_DEFAULT wine mt5setup.exe"
    MT5_EXE="$MT5_DEFAULT"
fi

# 4. Virtualenv
echo "[4/7] Creando entorno virtual Python..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "[OK] Virtualenv creado"
else
    echo "[OK] Virtualenv ya existe"
fi

# 5. Dependencias Python
echo "[5/7] Instalando dependencias Python..."
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "[OK] Dependencias instaladas"

# 6. Ollama
echo "[6/7] Verificando Ollama..."
if command -v ollama >/dev/null 2>&1; then
    echo "[OK] Ollama instalado"
    (ollama serve >/dev/null 2>&1 &) 2>/dev/null
    sleep 3
    if ollama pull phi3:mini >/dev/null 2>&1; then
        echo "[OK] Modelo phi3:mini listo"
    else
        echo "[WARN] No se pudo descargar phi3:mini"
    fi
else
    echo "[WARN] Ollama no instalado. Descarga desde https://ollama.ai"
fi

# 7. Directorios y config
echo "[7/7] Creando estructura de directorios..."
mkdir -p data/raw data/processed models/checkpoints models/reports logs/trades logs/backtest logs/system config

[ -f "config/runtime_params.json" ] || echo "{}" > config/runtime_params.json
[ -f "config/optimized_params.json" ] || echo "{}" > config/optimized_params.json

# .env
if [ ! -f ".env" ]; then
    cat > .env <<EOF
TRADING_MODE=demo
MT5_LOGIN=106049158
MT5_PASSWORD=r%;Qo)O8
MT5_SERVER=FBS-Demo
DISPLAY=:99
WINEPREFIX=$WINEPREFIX_DEFAULT
MT5_EXE=$MT5_EXE
MT5_BRIDGE_HOST=localhost
MT5_BRIDGE_PORT=8001
EOF
    echo "[OK] .env creado"
else
    echo "[OK] .env ya existe"
fi

# Verificar imports
python -c "from config import settings, constants; from utils import logger, display, notifier; print('  Imports Python: OK')"

echo
echo "===================================================="
echo "  Instalacion completada!"
echo "===================================================="
echo
echo "  PROXIMOS PASOS (trading en vivo con MT5 real):"
echo "  1. Instala MT5 dentro de Wine (ver aviso arriba si falta)."
echo "  2. Levanta el display virtual: Xvfb :99 -screen 0 1024x768x16 &"
echo "  3. Inicia sesion en MT5 (bajo Wine) con tus credenciales FBS."
echo "  4. Corre el servidor puente mt5linux con el Python de Wine, ej:"
echo "       WINEPREFIX=$WINEPREFIX_DEFAULT wine python -m mt5linux <ruta_python_windows>"
echo "     (ver README.md seccion 'MT5 en Linux' para el detalle completo)."
echo "  5. Ejecuta: source .venv/bin/activate && python main.py"
echo
echo "  ALTERNATIVA sin MT5 (entrenamiento/backtest con yfinance):"
echo "    ./run_test.sh"
echo
