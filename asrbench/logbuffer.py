"""Ring buffer + file persistente per observability.

Ogni riga: HH:MM:SS.mmm [tag] messaggio
Tag usati: sys, ui, rec, load, batch, live.
Il file resta in logs/ per rilettura post-sessione.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

MAX_LINES = 500
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = LOG_DIR / f"asr-bench-{RUN_ID}.log"

_lock = threading.Lock()
_lines: deque = deque(maxlen=MAX_LINES)
_fh = open(LOG_FILE, "a", encoding="utf-8")


def _write(tag: str, msg: str) -> str:
    ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
    line = f"{ts} [{tag}] {msg}"
    with _lock:
        _lines.append(line)
        try:
            _fh.write(line + "\n")
            _fh.flush()
        except Exception:
            pass
    print(line, flush=True)
    return line


def log(msg: str) -> str:
    """Compat: accetta '[tag] resto' oppure testo libero (tag=junk)."""
    msg = msg.strip()
    if msg.startswith("["):
        end = msg.find("]")
        if end > 1:
            return _write(msg[1:end], msg[end + 1 :].strip())
    return _write("log", msg)


def event(tag: str, msg: str) -> str:
    return _write(tag, msg)


def tail(n: int = 60) -> str:
    with _lock:
        lines = list(_lines)[-n:]
    return "\n".join(lines)


def clear() -> str:
    with _lock:
        _lines.clear()
    return ""


def path() -> str:
    return str(LOG_FILE)


_write("sys", f"session start run={RUN_ID} file={LOG_FILE.name}")
