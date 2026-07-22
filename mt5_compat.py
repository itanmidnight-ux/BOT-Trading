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


class _LazyMT5Linux:
    """Proxy perezoso sobre el cliente mt5linux.

    El cliente real abre la conexion TCP al servidor (que corre en el Python
    de Wine) en su constructor. Instanciarlo al importar este modulo hacia
    imposible siquiera importar el codebase (o correr los tests) sin ese
    servidor levantado; por eso la conexion se difiere al primer uso real."""

    def __init__(self, host: str, port: int):
        self._client = None
        self._host = host
        self._port = port

    def __getattr__(self, name):
        if self._client is None:
            from mt5linux import MetaTrader5 as _MT5LinuxClient  # type: ignore
            try:
                self._client = _MT5LinuxClient(host=self._host, port=self._port)
            except Exception as e:
                raise RuntimeError(
                    f"No se pudo conectar al servidor mt5linux en {self._host}:{self._port} ({e}). "
                    "Levanta el servidor dentro del Python de Wine antes de operar: "
                    "wine python -m mt5linux <ruta_python_wine> (ver README.md)."
                ) from e
        return getattr(self._client, name)


try:
    import MetaTrader5 as _mt5  # type: ignore
    BACKEND = "native"
except ImportError:
    try:
        import mt5linux  # noqa: F401  (solo se verifica que el paquete exista)
        _mt5 = _LazyMT5Linux(
            host=os.getenv("MT5_BRIDGE_HOST", "localhost"),
            port=int(os.getenv("MT5_BRIDGE_PORT", "18812")),
        )
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
