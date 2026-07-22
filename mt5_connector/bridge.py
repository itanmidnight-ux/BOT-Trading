"""
Bridge de archivos Python <-> expert_advisor.mq5.

Via alternativa de ejecucion (activada con USE_EA_BRIDGE=True) para brokers/
setups donde las ordenes deben originarse desde un EA dentro del terminal en
vez de la API python directa. Protocolo deliberadamente simple (texto plano
key=value) porque MQL5 no trae un parser JSON nativo en el lenguaje base.

Layout dentro de <Terminal Common Data Folder>/Files/bot_bridge/:
    commands/   Python escribe *.cmd (uno por orden). El EA los procesa y borra.
    acks/       El EA escribe <id>.ack con el resultado de cada comando.
    status/     El EA escribe status.status cada N ticks con cuenta + posiciones.

Escritura atomica: siempre se escribe a un archivo temporal `.tmp` y se
renombra al nombre final (`os.replace`), para que el EA nunca lea un archivo
a medio escribir. El EA replica el mismo patron para `status.status`.
"""
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


class BridgeTimeoutError(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, base_path: Optional[str] = None):
        root = Path(base_path or config.MT5_COMMON_FILES_PATH)
        if not base_path and not config.MT5_COMMON_FILES_PATH:
            raise ValueError(
                "MT5_COMMON_FILES_PATH no esta configurado. Es la carpeta "
                "'Common/Files' del terminal MT5 (visible en MT5 > Archivo > "
                "Abrir carpeta de datos > .. > Common > Files)."
            )
        self.bridge_dir = root / config.BRIDGE_SUBDIR
        self.commands_dir = self.bridge_dir / "commands"
        self.acks_dir = self.bridge_dir / "acks"
        self.status_dir = self.bridge_dir / "status"
        for d in (self.commands_dir, self.acks_dir, self.status_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Escritura de comandos
    # ------------------------------------------------------------------
    def _write_command(self, fields: Dict[str, str]) -> str:
        cmd_id = uuid.uuid4().hex
        fields["id"] = cmd_id
        fields["magic"] = str(config.BRIDGE_MAGIC)

        tmp_path = self.commands_dir / f"{cmd_id}.tmp"
        final_path = self.commands_dir / f"{cmd_id}.cmd"
        body = "\n".join(f"{k}={v}" for k, v in fields.items())
        tmp_path.write_text(body, encoding="ascii")
        os.replace(tmp_path, final_path)  # rename atomico
        return cmd_id

    def _wait_for_ack(self, cmd_id: str) -> Dict[str, str]:
        ack_path = self.acks_dir / f"{cmd_id}.ack"
        deadline = time.time() + config.BRIDGE_POLL_TIMEOUT_SEC
        while time.time() < deadline:
            if ack_path.exists():
                content = ack_path.read_text(encoding="ascii", errors="ignore")
                ack_path.unlink(missing_ok=True)
                return dict(line.split("=", 1) for line in content.splitlines() if "=" in line)
            time.sleep(config.BRIDGE_POLL_INTERVAL_SEC)
        raise BridgeTimeoutError(f"Sin ack del EA para comando {cmd_id} tras {config.BRIDGE_POLL_TIMEOUT_SEC}s")

    # ------------------------------------------------------------------
    # API publica de comandos
    # ------------------------------------------------------------------
    def send_open(self, direction: int, volume: float, sl: float, tp: Optional[float], comment: str = "") -> Dict:
        cmd_id = self._write_command({
            "type": "OPEN",
            "symbol": config.SYMBOL,
            "direction": str(direction),
            "volume": f"{volume:.2f}",
            "sl": f"{sl:.5f}",
            "tp": f"{tp:.5f}" if tp is not None else "0",
            "comment": comment[:31].replace("=", "-").replace("\n", " "),
        })
        return self._wait_for_ack(cmd_id)

    def send_close(self, ticket: int, volume: Optional[float] = None, comment: str = "") -> Dict:
        cmd_id = self._write_command({
            "type": "CLOSE",
            "ticket": str(ticket),
            "volume": f"{volume:.2f}" if volume is not None else "0",
            "comment": comment[:31].replace("=", "-").replace("\n", " "),
        })
        return self._wait_for_ack(cmd_id)

    def send_modify(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> Dict:
        cmd_id = self._write_command({
            "type": "MODIFY",
            "ticket": str(ticket),
            "sl": f"{sl:.5f}" if sl is not None else "-1",
            "tp": f"{tp:.5f}" if tp is not None else "-1",
        })
        return self._wait_for_ack(cmd_id)

    # ------------------------------------------------------------------
    # Lectura de estado publicado por el EA
    # ------------------------------------------------------------------
    def read_status(self) -> Optional[Dict]:
        status_path = self.status_dir / "status.status"
        if not status_path.exists():
            return None
        content = status_path.read_text(encoding="ascii", errors="ignore")
        lines = content.splitlines()
        result: Dict[str, object] = {"positions": []}
        positions: List[Dict[str, str]] = []
        for line in lines:
            if line.startswith("POS|"):
                parts = line[4:].split(",")
                keys = ["ticket", "type", "volume", "price_open", "sl", "tp", "profit"]
                positions.append(dict(zip(keys, parts)))
            elif "=" in line:
                k, v = line.split("=", 1)
                result[k] = v
        result["positions"] = positions
        return result
