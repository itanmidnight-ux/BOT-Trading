"""
data_updater.py
Gestiona actualizaciones incrementales de datos históricos para múltiples
pares e instrumentos, y decide cuándo re-entrenar el modelo.

Thread-safety: todos los accesos a los contadores internos están protegidos
por un threading.Lock para permitir su uso desde múltiples hilos (streaming
+ loop de entrenamiento).
"""

from __future__ import annotations

import threading
from typing import Optional

from utils.logger import get_logger
from config import settings
from core.data_downloader import download_historical, load_local

log = get_logger("data_updater")


class DataUpdater:
    """
    Descarga y mantiene actualizados los datos históricos para una lista de
    símbolos y timeframes.

    Parameters
    ----------
    symbols    : Lista de símbolos, p.ej. ["EURUSD", "XAUUSD"].
    timeframes : Lista de strings de TF, p.ej. ["M1", "M5", "M15"].
    """

    def __init__(self, symbols: list[str], timeframes: list[str]) -> None:
        self.symbols = list(symbols)
        self.timeframes = list(timeframes)

        # Contadores de velas nuevas acumuladas desde el último reentrenamiento
        # clave: symbol (el reentrenamiento es por símbolo, no por TF)
        self._new_candles: dict[str, int] = {s: 0 for s in self.symbols}

        # Marca de si el símbolo necesita reentrenamiento
        self._retrain_flag: dict[str, bool] = {s: False for s in self.symbols}

        # Lock para acceso thread-safe a los contadores
        self._lock = threading.Lock()

        log.info(
            "DataUpdater inicializado. Símbolos: %s | TFs: %s",
            self.symbols, self.timeframes,
        )

    # ── API pública ───────────────────────────────────────────────────────

    def update_all(self) -> dict[str, dict[str, int]]:
        """
        Actualiza datos para todos los symbol/TF configurados.

        Retorna un dict con la estructura:
            {
                "EURUSD": {"M1": 42, "M5": 8, "M15": 3},
                "XAUUSD": {"M1": 37, ...},
                ...
            }
        donde el valor es el número de velas nuevas añadidas en cada TF.
        """
        results: dict[str, dict[str, int]] = {}

        for symbol in self.symbols:
            results[symbol] = {}
            symbol_new_total = 0

            for tf in self.timeframes:
                try:
                    added = self.update_one(symbol, tf)
                    results[symbol][tf] = added
                    # Acumular velas del TF principal (M1) para trigger reentrenamiento
                    if tf == settings.TIMEFRAME_MAIN:
                        symbol_new_total += added
                except Exception as exc:
                    log.error(
                        "Error al actualizar %s %s: %s", symbol, tf, exc
                    )
                    results[symbol][tf] = 0

            # Actualizar contador global para el símbolo
            if symbol_new_total > 0:
                with self._lock:
                    self._new_candles[symbol] += symbol_new_total
                    if self._new_candles[symbol] >= settings.RETRAIN_EVERY_N_CANDLES:
                        self._retrain_flag[symbol] = True
                        log.info(
                            "[%s] Umbral de reentrenamiento alcanzado: "
                            "%d velas nuevas acumuladas (umbral=%d).",
                            symbol,
                            self._new_candles[symbol],
                            settings.RETRAIN_EVERY_N_CANDLES,
                        )

        return results

    def update_one(self, symbol: str, timeframe_str: str) -> int:
        """
        Actualiza datos para un único par symbol/TF.

        Descarga velas nuevas desde la última vela guardada hasta el presente.
        Retorna el número de velas nuevas añadidas al CSV local.
        """
        # Contar velas antes de la actualización
        count_before = self.get_candle_count(symbol, timeframe_str)

        # download_historical con comportamiento incremental si hay CSV
        try:
            df = download_historical(
                symbol=symbol,
                timeframe_str=timeframe_str,
                n_candles=settings.MIN_CANDLES_TO_TRAIN,
            )
        except Exception as exc:
            log.error(
                "update_one(%s, %s) falló: %s", symbol, timeframe_str, exc
            )
            raise

        count_after = len(df)
        added = max(0, count_after - count_before)

        if added > 0:
            log.info(
                "[%s %s] Actualización completa: +%d velas nuevas (total=%d).",
                symbol, timeframe_str, added, count_after,
            )
        else:
            log.debug(
                "[%s %s] Sin velas nuevas (total=%d).",
                symbol, timeframe_str, count_after,
            )

        return added

    def get_candle_count(self, symbol: str, timeframe_str: str) -> int:
        """
        Retorna el número de velas actualmente en el CSV local.
        Retorna 0 si el fichero no existe.
        """
        df = load_local(symbol, timeframe_str)
        if df is None:
            return 0
        return len(df)

    def should_retrain(self, symbol: str) -> bool:
        """
        Retorna True si el símbolo ha acumulado al menos
        settings.RETRAIN_EVERY_N_CANDLES velas nuevas desde el último
        reentrenamiento.

        Thread-safe.
        """
        with self._lock:
            return self._retrain_flag.get(symbol, False)

    def mark_retrained(self, symbol: str) -> None:
        """
        Resetea el contador de velas nuevas y el flag de reentrenamiento
        para el símbolo indicado.

        Debe llamarse después de completar un ciclo de entrenamiento.
        Thread-safe.
        """
        with self._lock:
            prev = self._new_candles.get(symbol, 0)
            self._new_candles[symbol] = 0
            self._retrain_flag[symbol] = False
            log.info(
                "[%s] Contador de reentrenamiento reseteado "
                "(tenía %d velas acumuladas).",
                symbol, prev,
            )

    def get_new_candles_count(self, symbol: str) -> int:
        """
        Retorna cuántas velas nuevas se han acumulado para el símbolo
        desde el último reentrenamiento.
        Thread-safe.
        """
        with self._lock:
            return self._new_candles.get(symbol, 0)

    def status_report(self) -> dict[str, dict]:
        """
        Retorna un resumen del estado actual de todos los símbolos y TFs.

        Formato:
            {
                "EURUSD": {
                    "candles": {"M1": 100000, "M5": 20000, ...},
                    "new_candles_since_retrain": 342,
                    "should_retrain": False,
                },
                ...
            }
        """
        report: dict[str, dict] = {}
        for symbol in self.symbols:
            candles: dict[str, int] = {}
            for tf in self.timeframes:
                candles[tf] = self.get_candle_count(symbol, tf)
            report[symbol] = {
                "candles": candles,
                "new_candles_since_retrain": self.get_new_candles_count(symbol),
                "should_retrain": self.should_retrain(symbol),
            }
        return report
