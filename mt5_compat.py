"""
Punto unico de import para la API de MetaTrader5.

- En Windows (o Wine con el paquete instalado en el Python del propio Wine),
  usa el paquete nativo `MetaTrader5`.
- En Linux nativo (sin Wine en el mismo interprete), cae al cliente del puente
  `mt5linux`, que habla con un servidor corriendo dentro del Python de Wine.
- Si ninguno esta disponible (p.ej. este sandbox de desarrollo, o corriendo los
  tests unitarios), expone un stub que permite importar el resto del codebase
  sin romper, y solo falla con un mensaje claro si algo intenta *usarlo*.

Todo el resto del proyecto debe importar `from mt5_compat import mt5` y nunca
`import MetaTrader5` directamente, para que este fallback funcione en todos lados.
"""
import logging
import os

logger = logging.getLogger(__name__)

_mt5 = None
BACKEND = None

try:
    import MetaTrader5 as _mt5  # type: ignore
    BACKEND = "native"
except ImportError:
    try:
        from mt5linux import MetaTrader5 as _MT5LinuxClient  # type: ignore
        _host = os.getenv("MT5_BRIDGE_HOST", "localhost")
        _port = int(os.getenv("MT5_BRIDGE_PORT", "18812"))
        _mt5 = _MT5LinuxClient(host=_host, port=_port)
        BACKEND = "mt5linux"
    except ImportError:
        _mt5 = None
        BACKEND = None


class _MT5Unavailable:
    """Stub para cuando no hay backend MT5 real disponible en el proceso."""

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise RuntimeError(
                "MetaTrader5 no esta disponible en este entorno (ni nativo ni via "
                "puente mt5linux). Instala el paquete `MetaTrader5` o levanta el "
                "servidor mt5linux antes de usar mt5_connector."
            )
        return _raise


if _mt5 is None:
    logger.warning("Backend MT5 no disponible: usando stub (solo permite importar, no operar).")
mt5 = _mt5 if _mt5 is not None else _MT5Unavailable()
