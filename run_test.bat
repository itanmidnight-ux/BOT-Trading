@echo off
cd /d "%~dp0"
echo ====================================================
echo   BOT-Trading v4.1 -- MODO TEST (sin MT5)
echo   Backtest completo + LLM con datos yfinance
echo ====================================================
echo.
call .venv\Scripts\activate.bat
python main.py --test
pause
