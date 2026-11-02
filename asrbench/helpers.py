"""Helper puri: nomi, VAD, scelte dropdown."""
from __future__ import annotations

import numpy as np

from asrbench import state
from asrbench.registry import MODELS


def short(mid: str) -> str:
    """Ultimo segmento dell'id HF, per log compatti."""
    return mid.split("/")[-1] if mid else "?"


def chunk_is_speech(wav: np.ndarray) -> bool:
    """RMS sopra soglia. Scala auto: peak > 1.5 = int16."""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 0:
        return False
    scale = 32768.0 if peak > 1.5 else 1.0
    rms = float(np.sqrt(np.mean((wav / scale) ** 2)))
    return rms > state.VAD_RMS


def sname(mid: str) -> str:
    """Nome corto per UI (Granite, Qwen3, Whisper, Nemotron)."""
    return state.SHORT.get(mid, short(mid))


def default_detail(mid: str) -> str:
    """Riga params | size dalla scheda registry."""
    m = next((x for x in MODELS if x.id == mid), None)
    return f"{m.params} | {m.size}" if m else ""


def model_choices():
    """Coppie (etichetta, id) per il dropdown. Solo modelli wired."""
    return [(f"{m.name} [{m.streaming_type}]", m.id) for m in MODELS if m.id in state.ADAPTERS]
