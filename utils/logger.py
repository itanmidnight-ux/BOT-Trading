import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

_BASE = Path(__file__).parent.parent
_LOG_DIR = _BASE / "logs" / "system"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_FMT = "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured: set = set()

def get_logger(name: str) -> logging.Logger:
    if name in _configured:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
        logger.addHandler(sh)

        log_file = _LOG_DIR / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
        fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
        logger.addHandler(fh)

    _configured.add(name)
    return logger
