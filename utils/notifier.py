from utils.logger import get_logger

_log = get_logger("notifier")

TRADE_OPEN    = "TRADE_OPEN"
TRADE_CLOSE   = "TRADE_CLOSE"
RISK_EVENT    = "RISK_EVENT"
PHASE_CHANGE  = "PHASE_CHANGE"
MODEL_UPDATE  = "MODEL_UPDATE"
TRAINING_DONE = "TRAINING_DONE"
ERROR         = "ERROR"

def notify(event_type: str, data: dict):
    if event_type == TRADE_OPEN:
        msg = (f"[>>>] {data['symbol']} {data['direction']} "
               f"{data['lots']:.2f}lots @ {data['entry']:.5f} | "
               f"SL:{data['sl']:.5f} TP1:{data['tp1']:.5f}")
        print(f"\n{msg}")
        _log.info(msg)

    elif event_type == TRADE_CLOSE:
        sign = "WIN" if data['pnl'] >= 0 else "LOSS"
        arrow = "[+]" if data['pnl'] >= 0 else "[-]"
        msg = (f"{arrow} {sign} {data['symbol']} {data['direction']} | "
               f"{data['pips']:+.1f}pips | ${data['pnl']:+.2f} | {data['reason']}")
        print(f"\n{msg}")
        _log.info(msg)

    elif event_type == RISK_EVENT:
        msg = f"[!!!] RIESGO: {data['reason']}"
        print(f"\n{'!' * 54}")
        print(msg)
        print(f"{'!' * 54}")
        _log.warning(msg)

    elif event_type == PHASE_CHANGE:
        msg = f"[***] FASE {data['phase']} ACTIVADA — capital ${data['capital']:.2f} | {data['symbol']} habilitado"
        print(f"\n{msg}")
        _log.info(msg)

    elif event_type == MODEL_UPDATE:
        msg = (f"[*] Modelo {data['symbol']} actualizado. "
               f"F1:{data.get('old_f1', 0):.3f}→{data.get('new_f1', 0):.3f} | "
               f"WR:{data.get('win_rate', 0):.1%}")
        print(f"\n{msg}")
        _log.info(msg)

    elif event_type == TRAINING_DONE:
        wr = data.get('win_rate', 0)
        ready = wr >= 0.59
        status = "LISTO PARA LIVE" if ready else "NO ALCANZÓ 59% WR"
        msg = f"[TRAINING] {status} | WR:{wr:.1%} | PF:{data.get('profit_factor', 0):.2f} | iter:{data.get('iteration', 0)}"
        print(f"\n{'*' * 54}")
        print(msg)
        print(f"{'*' * 54}")
        _log.info(msg)

    elif event_type == ERROR:
        msg = f"[ERROR] {data.get('message', 'Error desconocido')}"
        print(f"\n{msg}")
        _log.error(msg)
