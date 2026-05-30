@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ====================================================
echo    BOT-Trading v4.0 -- Instalador Windows
echo    MT5/FBS ^| Ensemble ML + Ollama LLM ^| Kelly 1:3000
echo ====================================================
echo.

:: 1. Python
echo [1/6] Verificando Python...
python --version >nul 2>&1 || (echo [ERROR] Python no encontrado. Instala Python 3.10+ desde python.org && exit /b 1)
python -c "import sys; v=sys.version_info; exit(0 if v.major==3 and v.minor>=8 else 1)" || (echo [ERROR] Requiere Python 3.8+ && exit /b 1)
echo [OK] Python OK

:: 2. MetaTrader 5
echo [2/6] Verificando MetaTrader 5...
set MT5_DEFAULT=C:\Program Files\MetaTrader 5\terminal64.exe
if exist "%MT5_DEFAULT%" (
    echo [OK] MT5 encontrado: %MT5_DEFAULT%
    set MT5_EXE=%MT5_DEFAULT%
) else (
    echo [WARN] MT5 no encontrado en ruta por defecto.
    echo   Descarga MT5 desde: https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
    echo   Instala MT5 y luego ejecuta este script nuevamente.
    set MT5_EXE=C:\Program Files\MetaTrader 5\terminal64.exe
)

:: 3. Virtualenv
echo [3/6] Creando entorno virtual Python...
if not exist ".venv" (
    python -m venv .venv
    echo [OK] Virtualenv creado
) else (
    echo [OK] Virtualenv ya existe
)

:: 4. Dependencias Python
echo [4/6] Instalando dependencias Python...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo [OK] Dependencias instaladas

:: 5. Ollama
echo [5/6] Verificando Ollama...
ollama --version >nul 2>&1 && (
    echo [OK] Ollama instalado
    ollama serve >nul 2>&1 &
    timeout /t 3 /nobreak >nul
    ollama pull phi3:mini >nul 2>&1 && echo [OK] Modelo phi3:mini listo || echo [WARN] No se pudo descargar phi3:mini
) || (
    echo [WARN] Ollama no instalado. Descarga desde https://ollama.ai
)

:: 6. Directorios y config
echo [6/6] Creando estructura de directorios...
for %%d in (data\raw data\processed models\checkpoints models\reports logs\trades logs\backtest logs\system config) do (
    if not exist "%%d" mkdir "%%d"
)

if not exist "config\runtime_params.json" echo {} > config\runtime_params.json
if not exist "config\optimized_params.json" echo {} > config\optimized_params.json

:: .env
if not exist ".env" (
    echo TRADING_MODE=demo > .env
    echo MT5_EXE=%MT5_EXE% >> .env
    echo MT5_LOGIN=106049158 >> .env
    echo MT5_PASSWORD=r%%;Qo)O8 >> .env
    echo MT5_SERVER=FBS-Demo >> .env
    echo [OK] .env creado
) else (
    echo [OK] .env ya existe
)

:: Verificar imports
python -c "from config import settings, constants; from utils import logger, display, notifier; print('  Imports Python: OK')"

echo.
echo ====================================================
echo   Instalacion completada!
echo ====================================================
echo.
echo   PROXIMOS PASOS:
echo   1. Abre MetaTrader 5
echo   2. Inicia sesion: login=106049158 server=FBS-Demo
echo   3. Deja MT5 abierto en segundo plano
echo   4. Ejecuta: .venv\Scripts\activate ^&^& python main.py
echo.
pause
