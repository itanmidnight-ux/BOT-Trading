"""
data_downloader.py
Descarga velas históricas desde MetaTrader 5 y las persiste en CSV.

Comportamiento:
- Si ya existe un CSV local se carga y solo se descargan las velas nuevas (incremental).
- La descarga desde MT5 se hace en chunks de CHUNK_SIZE velas para no saturar la API.
- Los timestamps MT5 (int Unix segundos) se convierten a datetime UTC.
- Se muestra una barra de progreso ASCII en stdout.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import MetaTrader5 as mt5

from utils.logger import get_logger
from config.constants import CANDLE_COLS
from config import settings

log = get_logger("data_downloader")

# ── Constantes ────────────────────────────────────────────────────────────────

CHUNK_SIZE = 50_000          # velas por solicitud a MT5
PROGRESS_WIDTH = 40          # ancho de la barra de progreso (caracteres)

# Mapeo de string → constante MT5
TF_MAP: dict[str, int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
}

# Segundos por timeframe (para cálculos de gaps y periodos)
TF_SECONDS: dict[str, int] = {
    "M1":  60,
    "M5":  300,
    "M15": 900,
    "H1":  3600,
    "H4":  14400,
}


# ── Helpers privados ──────────────────────────────────────────────────────────

def _csv_path(symbol: str, timeframe_str: str) -> Path:
    """Ruta canónica del CSV para el par symbol/TF."""
    settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    return settings.DATA_RAW_DIR / f"{symbol}_{timeframe_str}.csv"


def _render_progress(done: int, total: int, width: int = PROGRESS_WIDTH) -> str:
    """
    Devuelve una cadena tipo:
        [=====>    ] 45000/100000 velas
    """
    frac = min(done / total, 1.0) if total > 0 else 1.0
    filled = int(frac * width)
    bar_chars = "=" * filled
    if filled < width:
        bar_chars += ">"
    bar_chars = bar_chars.ljust(width)
    return f"[{bar_chars}] {done:,}/{total:,} velas"


def _print_progress(done: int, total: int) -> None:
    """Escribe la barra de progreso en stdout (sobreescribe la línea)."""
    line = _render_progress(done, total)
    sys.stdout.write(f"\r{line}")
    sys.stdout.flush()


def _rates_to_df(rates) -> pd.DataFrame:
    """
    Convierte el numpy recarray devuelto por MT5 a DataFrame limpio.
    Columnas resultantes: CANDLE_COLS.
    La columna 'time' queda como datetime64[ns, UTC].
    """
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # Asegurar que solo están las columnas estándar en el orden correcto
    existing = [c for c in CANDLE_COLS if c in df.columns]
    return df[existing].reset_index(drop=True)


def _fetch_chunk(symbol: str, tf_mt5: int, start_pos: int, count: int) -> Optional[pd.DataFrame]:
    """
    Descarga hasta *count* velas comenzando en el offset *start_pos*
    (0 = vela más reciente).
    Retorna DataFrame o None si MT5 no devuelve datos.
    """
    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, start_pos, count)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        log.warning(
            "copy_rates_from_pos(%s, pos=%d, count=%d) devolvió vacío. Error MT5: %s",
            symbol, start_pos, count, err,
        )
        return None
    return _rates_to_df(rates)


def _fetch_incremental(
    symbol: str,
    tf_mt5: int,
    tf_str: str,
    since_dt: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Descarga velas desde *since_dt* hasta ahora usando copy_rates_range.
    Retorna DataFrame con las velas nuevas, o None si no hay nada.
    """
    import datetime as dt

    now_utc = pd.Timestamp.utcnow().replace(tzinfo=None)
    since_naive = since_dt.tz_convert("UTC").replace(tzinfo=None)

    # Añadir 1 período para no re-descargar la última vela ya guardada
    delta = pd.Timedelta(seconds=TF_SECONDS.get(tf_str, 60))
    since_naive = (pd.Timestamp(since_naive) + delta).to_pydatetime()

    rates = mt5.copy_rates_range(
        symbol,
        tf_mt5,
        since_naive,
        now_utc,
    )
    if rates is None or len(rates) == 0:
        return None
    return _rates_to_df(rates)


# ── API pública ───────────────────────────────────────────────────────────────

def load_local(symbol: str, timeframe_str: str) -> Optional[pd.DataFrame]:
    """
    Carga el CSV local de symbol/TF.
    Retorna DataFrame (con 'time' como datetime UTC) o None si el fichero
    no existe o está vacío.
    """
    path = _csv_path(symbol, timeframe_str)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["time"])
        if df.empty:
            return None
        # Garantizar timezone UTC
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")
        log.debug("CSV cargado: %s (%d filas)", path, len(df))
        return df
    except Exception as exc:
        log.error("Error al leer CSV %s: %s", path, exc)
        return None


def get_missing_count(symbol: str, timeframe_str: str, target_n: int) -> int:
    """
    Retorna cuántas velas faltan en el CSV local para alcanzar *target_n*.
    Si no existe CSV retorna target_n completo.
    """
    df = load_local(symbol, timeframe_str)
    if df is None:
        return target_n
    current = len(df)
    return max(0, target_n - current)


def download_historical(
    symbol: str,
    timeframe_str: str,
    n_candles: int = 100_000,
) -> pd.DataFrame:
    """
    Descarga hasta *n_candles* velas históricas para *symbol* en *timeframe_str*.

    Estrategia:
    1. Si existe CSV local: carga los datos existentes y solo descarga
       las velas nuevas (desde la última vela hasta ahora).
    2. Si no existe CSV: descarga desde MT5 en chunks de CHUNK_SIZE velas,
       mostrando progreso en terminal.

    Guarda el resultado en data/raw/{symbol}_{timeframe_str}.csv y retorna
    el DataFrame completo con columnas CANDLE_COLS.

    Raises:
        ValueError: si timeframe_str no está en TF_MAP.
        RuntimeError: si MT5 no devuelve ningún dato.
    """
    if timeframe_str not in TF_MAP:
        raise ValueError(
            f"Timeframe '{timeframe_str}' no reconocido. "
            f"Opciones válidas: {list(TF_MAP.keys())}"
        )

    tf_mt5 = TF_MAP[timeframe_str]
    csv_path = _csv_path(symbol, timeframe_str)

    # ── Caso 1: CSV existente → actualización incremental ──────────────────
    existing_df = load_local(symbol, timeframe_str)
    if existing_df is not None and not existing_df.empty:
        last_ts = existing_df["time"].max()
        log.info(
            "[%s %s] CSV local: %d velas. Última: %s. Descargando incrementales…",
            symbol, timeframe_str, len(existing_df), last_ts,
        )
        new_df = _fetch_incremental(symbol, tf_mt5, timeframe_str, last_ts)

        if new_df is not None and not new_df.empty:
            # Eliminar duplicados por si la última vela ya estaba
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
            log.info(
                "[%s %s] +%d velas nuevas → total %d.",
                symbol, timeframe_str, len(new_df), len(combined),
            )
        else:
            log.info("[%s %s] Sin velas nuevas disponibles.", symbol, timeframe_str)
            combined = existing_df

        # Limitar a las n_candles más recientes
        if len(combined) > n_candles:
            combined = combined.tail(n_candles).reset_index(drop=True)

        _save_csv(combined, csv_path, symbol, timeframe_str)
        return combined

    # ── Caso 2: Sin CSV → descarga completa por chunks ─────────────────────
    log.info(
        "[%s %s] Sin datos locales. Descargando %d velas en chunks de %d…",
        symbol, timeframe_str, n_candles, CHUNK_SIZE,
    )

    chunks: list[pd.DataFrame] = []
    total_fetched = 0
    start_pos = 0

    while total_fetched < n_candles:
        remaining = n_candles - total_fetched
        fetch_now = min(CHUNK_SIZE, remaining)

        chunk_df = _fetch_chunk(symbol, tf_mt5, start_pos, fetch_now)
        if chunk_df is None or chunk_df.empty:
            log.warning(
                "[%s %s] No se obtuvieron datos en pos=%d. Deteniendo descarga.",
                symbol, timeframe_str, start_pos,
            )
            break

        chunks.append(chunk_df)
        fetched_this = len(chunk_df)
        total_fetched += fetched_this
        start_pos += fetched_this

        _print_progress(total_fetched, n_candles)

        # MT5 devolvió menos de lo pedido → no hay más velas disponibles
        if fetched_this < fetch_now:
            log.debug(
                "[%s %s] MT5 devolvió %d < %d → fin del histórico disponible.",
                symbol, timeframe_str, fetched_this, fetch_now,
            )
            break

        # Pequeña pausa para no saturar MT5
        time.sleep(0.05)

    # Salto de línea tras la barra de progreso
    sys.stdout.write("\n")
    sys.stdout.flush()

    if not chunks:
        raise RuntimeError(
            f"MT5 no devolvió ninguna vela para {symbol} {timeframe_str}. "
            "Verifica la conexión y que el símbolo esté disponible."
        )

    # Combinar chunks (vienen en orden descendente de tiempo, revertir)
    full_df = pd.concat(chunks, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    log.info(
        "[%s %s] Descarga completa: %d velas (rango %s → %s).",
        symbol, timeframe_str, len(full_df),
        full_df["time"].min(), full_df["time"].max(),
    )

    _save_csv(full_df, csv_path, symbol, timeframe_str)
    return full_df


# ── Persistencia ──────────────────────────────────────────────────────────────

def _save_csv(df: pd.DataFrame, path: Path, symbol: str, timeframe_str: str) -> None:
    """Guarda DataFrame en CSV. 'time' se escribe en formato ISO 8601 UTC."""
    try:
        df.to_csv(path, index=False)
        log.info(
            "[%s %s] CSV guardado: %s (%d filas).",
            symbol, timeframe_str, path, len(df),
        )
    except Exception as exc:
        log.error("Error al guardar CSV %s: %s", path, exc)
        raise
