"""asr-bench: benchmark live di ASR italiano. Entry point sottile: init + UI."""
from __future__ import annotations

from asrbench import state
from asrbench.logbuffer import event, path as log_path
from asrbench.status import model_status_init
from asrbench.trials import write_session_files
from asrbench.ui import build_demo

model_status_init()
write_session_files()
event("sys", f"app init modelli={len([m for m in state.MODEL_STATUS])} log={log_path()}")

demo = build_demo()

if __name__ == "__main__":
    demo.launch()  # su HF Space: bind 0.0.0.0:7860 automatico
