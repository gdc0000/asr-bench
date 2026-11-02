"""Stato globale condiviso: adapter registrati, modello attivo, sessione, contatori."""
from __future__ import annotations

from asrbench.adapters.granite import GraniteAdapter
from asrbench.adapters.nemotron import NemotronAdapter
from asrbench.adapters.qwen import QwenAdapter
from asrbench.adapters.whisper import WhisperAdapter
from asrbench.core.logbuffer import RUN_ID

ADAPTERS = {
    "ibm-granite/granite-speech-5.0-470m-turboctc": GraniteAdapter,
    "Qwen/Qwen3-ASR-0.6B": QwenAdapter,
    "openai/whisper-small": WhisperAdapter,
    "nvidia/nemotron-3.5-asr-streaming-0.6b": NemotronAdapter,
}

DASH = "—"
SHORT = {
    "ibm-granite/granite-speech-5.0-470m-turboctc": "Granite",
    "Qwen/Qwen3-ASR-0.6B": "Qwen3",
    "openai/whisper-small": "Whisper",
    "nvidia/nemotron-3.5-asr-streaming-0.6b": "Nemotron",
}
VAD_RMS = 0.02  # soglia voce su scala -1..1
NEMOTRON_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

CURRENT = {"adapter": None, "id": None}
SELECTED = {"model": None}
MODEL_STATUS = {}
TRIAL_SEQ = 0
TRIAL_ACTIVE = False
VERBOSE = {"live_chunks": False}
SESSION: list = []

CSV_PATH = f"logs/session-{RUN_ID}.csv"
JSONL_PATH = f"logs/session-{RUN_ID}.jsonl"
CSV_COLS = ["prova", "ora", "modello", "audio_s", "gt", "testo",
            "wer", "cer", "latenza_primo_testo_s", "inferenza_s", "rtf", "revisioni"]
