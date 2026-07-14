"""
mt5_compat.py
Access point for the MetaTrader5 API on Linux.

The `MetaTrader5` Python package is a Windows-only binding (it talks to a
running terminal via a local Windows IPC channel), so it cannot be installed
on Linux. The real terminal + its Python API run inside Wine, and this
process connects to them through `mt5linux`
(https://github.com/lucas-campagna/mt5linux), which exposes the exact same
class/attribute surface (mt5.initialize(), mt5.TIMEFRAME_M1, ...) over RPyC.

Every other module should do `from core.mt5_compat import mt5` instead of
`import MetaTrader5 as mt5`, so test suites that mock
`sys.modules["MetaTrader5"]` keep working unchanged.
"""
import os

try:
    import MetaTrader5 as mt5
except ImportError:
    from mt5linux import MetaTrader5 as _MT5Bridge

    mt5 = _MT5Bridge(
        host=os.environ.get("MT5_BRIDGE_HOST", "localhost"),
        port=int(os.environ.get("MT5_BRIDGE_PORT", "8001")),
    )

__all__ = ["mt5"]
