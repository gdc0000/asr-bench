"""asr-bench: benchmark live di ASR italiano. Entry point sottile: init + UI."""
from __future__ import annotations

from asrbench.core import state
from asrbench.core.logbuffer import event, path as log_path
from asrbench.engine.trials import write_session_files
from asrbench.ui.layout import build_demo
from asrbench.ui.status import model_status_init

model_status_init()
write_session_files()
event("sys", f"app init modelli={len([m for m in state.MODEL_STATUS])} log={log_path()}")

demo = build_demo()

if __name__ == "__main__":
    demo.launch()  # su HF Space: bind 0.0.0.0:7860 automatico
